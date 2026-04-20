"""
ID-MIM测试脚本：评估跨模态舰船ReID性能 + 可视化结果
"""
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import logging
from config import cfg
from datasets.make_dataloader_idmim import make_dataloader_idmim
from model.make_model_idmim import make_model_idmim
from utils.logger import setup_logger
from utils.metrics import evaluate_rank, compute_mutual_info_score
from utils.visualization import tsne_visualize, attention_heatmap, recon_visualize

def load_model(cfg, num_classes, num_cameras):
    """加载微调后的ID-MIM模型"""
    model = make_model_idmim(cfg, num_class=num_classes, camera_num=num_cameras, phase='finetune')

    # 加载测试权重
    if os.path.exists(cfg.TEST.WEIGHT):
        checkpoint = torch.load(cfg.TEST.WEIGHT, map_location='cpu')
        model.load_state_dict(checkpoint)
        logger = logging.getLogger("ID-MIM Test")
        logger.info(f"Loaded test weight from: {cfg.TEST.WEIGHT}")
    else:
        raise ValueError(f"Test weight not found: {cfg.TEST.WEIGHT}")

    model = model.cuda()
    model.eval()
    return model

def extract_features(model, data_loader, feat_norm=True):
    """提取数据集的身份特征（用于可视化/评估）"""
    features = []
    pids = []
    mods = []
    imgs = []

    with torch.no_grad():
        for batch_data in data_loader:
            img, pid, camid, viewid, mod = batch_data
            img = img.cuda()

            # 提取身份特征
            feat = model[0](img, mod.cuda())  # [B, 768]

            # 特征归一化
            if feat_norm:
                feat = torch.nn.functional.normalize(feat, dim=1)

            # 收集结果
            features.append(feat.cpu())
            pids.extend(pid.numpy())
            mods.extend(mod.numpy())
            imgs.append(img.cpu())

    # 拼接所有特征/标签
    features = torch.cat(features, dim=0)  # [N, 768]
    pids = np.array(pids)                  # [N]
    mods = np.array(mods)                  # [N]
    imgs = torch.cat(imgs, dim=0)          # [N, 3, H, W]

    return features, pids, mods, imgs



def main():
    """命令行参数解析 + 启动测试"""
    parser = argparse.ArgumentParser(description='ID-MIM Test for Ship ReID')
    parser.add_argument(
        "--config_file", default="configs/finetune_idmim.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # 加载配置
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 启动测试
    test_idmim(cfg)

if __name__ == '__main__':
    main()