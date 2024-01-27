import os
import sys
import random
import numpy as np
import os.path as osp
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import default_collate
from utils.utils import ExtRandomCrop

# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

class TrainDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = osp.join(root, "training")
        self.img_list = os.listdir(osp.join(self.root, "image_2"))
       
    def __getitem__(self, idx):
        filename = self.img_list[idx]
        img = Image.open(osp.join(self.root, 'image_2', filename))
        label = Image.open(osp.join(self.root, 'gt_image_2', filename.replace("_", "_road_")))

        # trans = ExtRandomCrop(size=(320, 500))
        # img,label = trans(img,label)

        if torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)

        brightness_factor = random.uniform(0.9, 1.1)
        img = TF.adjust_brightness(img, brightness_factor)

        scale_factor = random.uniform(0.5, 2)
        scale = (int(scale_factor * img.width), int(scale_factor * img.height))
        img = img.resize(scale, resample=Image.NEAREST)
        label = label.resize(scale, resample=Image.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)

        label = torch.from_numpy(np.array(label, dtype=np.int32)[...,2]) / 255

        return {"img": img, "label":label}

    def __len__(self) -> int:
        return self.img_list.__len__()
    
class ValDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = osp.join(root, "testing")
        self.img_list = os.listdir(osp.join(self.root, "image_2"))
        
    def __getitem__(self, idx):
        filename = self.img_list[idx]
        img = Image.open(osp.join(self.root, 'image_2', filename))
        label = Image.open(osp.join(self.root, 'gt_image_2', filename.replace("_", "_road_")))
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
        label = torch.from_numpy(np.array(label, dtype=np.int32)[...,2]) / 255
        
        return {"img": img, "label":label, "filename": osp.join(self.root, 'image_2', filename)}
    
    def __len__(self) -> int:
        return self.img_list.__len__()


class TestDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = osp.join(root, "testing")
        self.img_list = os.listdir(osp.join(self.root, "image_2"))
        
    def __getitem__(self, idx):
        filename = self.img_list[idx]
        img = Image.open(osp.join(self.root, 'image_2', filename))
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)

        return {"img": img, "filename": osp.join(self.root, 'image_2', filename)}
    
    def __len__(self) -> int:
        return self.img_list.__len__()

def collate(batch, target_size=(375, 1240)):
    for i in range(len(batch)):
        height, width = batch[i]["label"].shape
        pad_h, pad_w = target_size[0] - height, target_size[1] - width
        left_pad, up_pad = pad_w // 2, pad_h // 2
        pad = [left_pad, pad_w-left_pad, up_pad, pad_h-up_pad]
        batch[i]["img"] = F.pad(batch[i]["img"], pad=pad, value=0)
        batch[i]["label"] = F.pad(batch[i]["label"], pad=pad, value=-1)
    return default_collate(batch)

def train_dataloader(cfg):
    train_dataset = TrainDataset(root=cfg['datasets']['data_path'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], 
                              num_workers=cfg['training']['num_workers'], 
                              collate_fn=lambda *args: collate(*args, (cfg['training']['size'][0],cfg['training']['size'][1])),
                              shuffle=True, pin_memory=True)
    return train_loader

def val_dataloader(cfg):
    val_dataset = ValDataset(root=cfg['datasets']['data_path'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    return val_loader

def test_dataloader(cfg):
    test_dataset = TestDataset(root=cfg['datasets']['data_path'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    return test_loader

if __name__ == "__main__":
    train_dataset = TrainDataset()
    test_dataset = TestDataset()
    
    from torch.utils.data import DataLoader
    train_loader = train_dataloader()
    test_loader = test_dataloader()
    for batch in train_loader:
        print(batch["label"].max())
