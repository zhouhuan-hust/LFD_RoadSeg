import os
import sys
sys.path.append(os.getcwd())
import argparse
import warnings
warnings.filterwarnings('ignore')
from utils.load_config import load_yaml
from models._LFDRoadSeg import LFDRoadSeg

def get_args():
    parser = argparse.ArgumentParser(description='LFD_RoadSeg ultrafast road segmentation')
    parser.add_argument('--config', type=str, default='configs/LFD_RoadSeg.yaml', help='config file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    cfg = load_yaml(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['device']
    print(cfg)

    os.makedirs(cfg['outputs']['ckpt_path'], exist_ok=True)
    os.makedirs(cfg['outputs']['vis_path'], exist_ok=True)
    model = LFDRoadSeg(cfg)
    
    if cfg['training']['state']:
        model.train()
    if cfg['validating']['state']:
        model.val()  # for kittival
    if cfg['testing']['state']:
        model.test() # for kitti
    if cfg['eval_speed']['state']:   
        model.eval_fps()
