import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import time
from thop import profile
import subprocess
from torch.optim import SGD, Adam
from models.scheduler.lr_scheduler import LR_Scheduler
from models.loss.LFD_loss import FSOhemCELoss
from models.net.LFDRoadseg_core import LFD_RoadSeg
from utils.utils import AverageMeter


class LFDRoadSeg():
    def __init__(self, cfg):
        self.cfg = cfg
        self.scale_factor = cfg['training']['scale_factor']
        weight_model = ""
        if cfg['training']['state'] and cfg['training']['start_epoch'] > 0:
            weight_model = os.path.join(
                cfg['outputs']['ckpt_path'], 'model_epoch_{}.pth'.format(cfg['training']['start_epoch']))
            assert os.path.exists(weight_model), "checkpoint does not exist!"
        elif cfg['validating']['state']:
            weight_model = os.path.join(
                cfg['outputs']['ckpt_path'], 'model_epoch_{}.pth'.format(cfg['validating']['val_epoch']))
            assert os.path.exists(weight_model), "checkpoint does not exist!"
        elif cfg['testing']['state']:
            weight_model = os.path.join(
                cfg['outputs']['ckpt_path'], 'model_epoch_{}.pth'.format(cfg['testing']['test_epoch']))
            assert os.path.exists(weight_model), "checkpoint does not exist!"
      
        if cfg['datasets']['dataset_name'] == 'kitti':
            from datasets.kitti_dataloader import train_dataloader, test_dataloader
            self.train_loader = train_dataloader(cfg)
            self.test_loader = test_dataloader(cfg)
            self.model = LFD_RoadSeg(self.scale_factor)
        elif cfg['datasets']['dataset_name'] == 'kittival':
            from datasets.kitti_dataloader import train_dataloader, val_dataloader
            self.train_loader = train_dataloader(cfg)
            self.val_loader = val_dataloader(cfg)
            self.model = LFD_RoadSeg(self.scale_factor)

        if len(weight_model) > 0:
            param_dict = torch.load(weight_model, map_location='cpu')
            param_dict = {k.replace("module.", ""): param_dict.pop(k)  for k in list(param_dict.keys())}
            self.model.load_state_dict(param_dict)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.optimizer = SGD(self.model.parameters(), lr=cfg['training']['lr'], momentum=0.9, weight_decay=1e-4)
        self.scheduler = LR_Scheduler(self.optimizer, 0, 0, cfg['training']['start_epoch'], cfg['training']['max_epochs'], cfg['training']['lr'], 1e-5, len(self.train_loader))
        self.global_progress = tqdm(range(cfg['training']['start_epoch'], cfg['training']['max_epochs']), desc="Training")
        self.loss_func = FSOhemCELoss(ignore_index=-1)

    def train(self):
        for epoch in self.global_progress:
            local_progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg['training']['max_epochs']}", 
                                                disable=self.cfg['training']['hide_bar'], dynamic_ncols=True)
            ave_loss = AverageMeter()
            self.model = self.model.train()
            for i, batch in enumerate(local_progress):
                self.model.zero_grad()
                batch = {k:batch[k].cuda() for k in ["img", "label"]}
                y_pred = self.model(batch)
                loss = self.loss_func(y_pred, batch["label"])

                loss.backward()
                lr = self.scheduler.step()

                self.optimizer.step()
                ave_loss.update(loss.data.item())
                local_progress.set_postfix({"lr":lr, "loss": ave_loss.average()})
                if (epoch + 1) % 10 == 0:
                    torch.save(self.model.state_dict(),'{}/model_epoch_{}.pth'.format(self.cfg['outputs']['ckpt_path'], epoch+1))
    
    # for kiitival 
    def val(self):
        self.model.eval()
        for i,batch in enumerate(self.val_loader):
            print(i)
            assert batch["img"].shape[0] == 1
            batch["img"] = batch["img"].cuda()
            with torch.no_grad():
                y_pred = self.model(batch)
            y_pred = F.softmax(y_pred, dim=1)[:, 1]
            y_pred = y_pred.cpu().numpy()[0]
            y_pred = (y_pred * 255).astype(np.uint8)
            
            res_dir = self.cfg['outputs']['vis_path']
            val_path = self.cfg['datasets']['data_path']
            if not os.path.isdir(os.path.join(res_dir, "scoremap")):
                os.makedirs(os.path.join(res_dir, "scoremap"))
            if not os.path.isdir(os.path.join(res_dir, "bevpred")):
                os.makedirs(os.path.join(res_dir, "bevpred"))
            if not os.path.isdir(os.path.join(res_dir, "vis")) and self.cfg['validating']['vis']:
                os.makedirs(os.path.join(res_dir, "vis"))
            filename = batch["filename"][0].split("/")[-1]
            if self.cfg['validating']['vis']:
                visulization_road(batch["filename"][0], 
                        os.path.join(res_dir, "vis", filename), y_pred, color=(0, 0, 255))
            flag = cv2.imwrite(os.path.join(res_dir, "scoremap", filename.replace("_", "_road_")), y_pred)
            if not flag:
                raise IOError("save image fail")
        # evaluate metrics from a camera perspective view
        subprocess.run(["python", "utils/devkit_road/python/evaluateRoad_1.py", "%s/scoremap" % res_dir, "%s/testing" % val_path])
        if not os.path.isdir(os.path.join(val_path, "gt_image_2")):
            os.makedirs(os.path.join(val_path, "gt_image_2"))
            subprocess.run(["python", "utils/devkit_road/python/transform2BEV.py", "%s/testing/gt_image_2/*.png" % val_path, "%s/testing/calib" % val_path, "%s/gt_image_2" % val_path])
        # generate bird eye view prediction maps 
        subprocess.run(["python", "utils/devkit_road/python/transform2BEV.py", "%s/scoremap/*.png" % res_dir, "%s/testing/calib" % val_path, "%s/bevpred" % res_dir])
        # evaluate metrics from a bird eye view
        subprocess.run(["python", "utils/devkit_road/python/evaluateRoad_1.py", "%s/bevpred" % res_dir, "%s" % val_path])


    def test(self): 
        self.model.eval()
        for i,batch in enumerate(self.test_loader):
            print(i)
            assert batch["img"].shape[0] == 1
            batch["img"] = batch["img"].cuda()
            with torch.no_grad():
                y_pred = self.model(batch)
            y_pred = F.softmax(y_pred, dim=1)[:, 1]
            y_pred = y_pred.cpu().numpy()[0]
            y_pred = (y_pred * 255).astype(np.uint8)         
            res_dir = self.cfg['outputs']['vis_path']
            test_path = self.cfg['datasets']['data_path']
            if not os.path.isdir(os.path.join(res_dir, "scoremap")):
                os.makedirs(os.path.join(res_dir, "scoremap"))
            if not os.path.isdir(os.path.join(res_dir, "bevpred")):
                os.makedirs(os.path.join(res_dir, "bevpred"))
            if not os.path.isdir(os.path.join(res_dir, "vis")) and self.cfg['testing']['vis']:
                os.makedirs(os.path.join(res_dir, "vis"))
            filename = batch["filename"][0].split("/")[-1]
            if self.cfg['testing']['vis']:
                visulization_road(batch["filename"][0], 
                        os.path.join(res_dir, "vis", filename), y_pred, color=(0, 0, 255))
            flag = cv2.imwrite(os.path.join(res_dir, "scoremap", filename.replace("_", "_road_")), y_pred)
            if not flag:
                raise IOError("save image fail")
            # generate bird eye view prediction maps 
            subprocess.run(["python", "utils/devkit_road/python/transform2BEV.py", "%s/scoremap/*.png" % res_dir, "%s/testing/calib" % test_path, "%s/bevpred" % res_dir])
 
    def eval_fps(self):
        cudnn.benchmark = True

        self.model.eval()
        self.model = self.model.cuda()

        img_size = self.cfg['eval_speed']['size']
        input_size = (self.cfg['eval_speed']['batch_size'], self.cfg['eval_speed']['num_channels'], img_size[0], img_size[1])
        img = torch.randn(input_size, device='cuda:0')
        input = {'img':img}
        iteration = self.cfg['eval_speed']['iter']
        macs, params = profile(self.model, inputs=(input, ))
        print("macs:", macs,"params:", params)
        for _ in range(50):
            self.model(input)
             
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iteration):
            self.model(input)
            print("iteration {}/{}".format(_, iteration), end='\r')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start

        speed_time = elapsed_time / iteration * 1000
        fps = iteration / elapsed_time

        print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
        print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
        return speed_time, fps

def visulization_road(filepath, target_path, y_pred, color=[1, 64, 128]):
    img = cv2.imread(filepath)
    mask = y_pred > 127
    img[mask] = img[mask] * 0.7 +  np.array(color) * 0.3
    flag = cv2.imwrite(target_path, img)
    if not flag:
        raise IOError("save vis image fail")


if __name__ == "__main__":
    model = LFD_RoadSeg()
    model.eval()
    print(model)
    x = torch.rand(2, 3, 256, 256)
    pred = model({"img":x})
    print(pred.shape)