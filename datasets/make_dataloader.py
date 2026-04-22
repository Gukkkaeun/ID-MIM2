import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

from .HOSS import HOSS
from .CMship import CMship
from .KT_Boat import KT_Boat

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg


__factory = {
    'HOSS': HOSS,
    'CMship': CMship,
    'KT_Boat': KT_Boat,
}


def train_collate_fn(batch):
    # 数据集返回：img, pid, camid, viewid, img_size
    imgs, pids, camids, viewids, img_sizes = zip(*batch)
    
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    img_wh = torch.tensor(img_sizes, dtype=torch.int64)
    
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_wh


def train_pair_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    rgb_batch = [i[0] for i in batch]
    sar_batch = [i[1] for i in batch]
    batch = rgb_batch + sar_batch
    imgs, pids, camids, viewids, img_wh = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    img_wh = torch.tensor(img_wh, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids, img_wh


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_sizes = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    img_wh = torch.tensor(img_sizes, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids, img_wh



def make_dataloader(cfg):
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
    )

    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    num_cam = dataset.num_train_cams
    num_view = dataset.num_train_vids


    train_set = ImageDataset(dataset.train, train_transforms)
    if "triplet" in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print("DIST_TRAIN START")

            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
            )
    elif cfg.DATALOADER.SAMPLER == "softmax":
        print("using softmax sampler")
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        print("unsupported sampler! expected softmax or triplet but got {}".format(cfg.SAMPLER))


    train_set_pair = ImageDataset(dataset.train_pair, train_transforms, pair=True)
    if "triplet" in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print("DIST_TRAIN START")

            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train_pair, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_pair = DataLoader(
                train_set_pair,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_pair_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_pair = DataLoader(
                train_set_pair,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train_pair, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                collate_fn=train_pair_collate_fn,
            )
    elif cfg.DATALOADER.SAMPLER == "softmax":
        print("using softmax sampler")
        train_loader_pair = DataLoader(
            train_set_pair, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers, collate_fn=train_pair_collate_fn
        )
    else:
        print("unsupported sampler! expected softmax or triplet but got {}".format(cfg.SAMPLER))

    
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    train_loader_normal = DataLoader(
        train_set_normal, 
        batch_size = cfg.TEST.IMS_PER_BATCH, 
        shuffle = False, 
        num_workers = num_workers, 
        collate_fn = val_collate_fn
    )

    test_set = ImageDataset(dataset.test, val_transforms)
    test_loader = DataLoader(
        test_set, 
        batch_size = cfg.TEST.IMS_PER_BATCH, 
        shuffle = False, 
        num_workers = num_workers, 
        collate_fn = val_collate_fn
    )

    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
    return train_loader, train_loader_normal, train_loader_pair, test_loader, len(dataset.test), num_classes, num_cam, num_view



# # 测试代码：验证加载逻辑
# if __name__ == '__main__':
#     try:
#         train_loader, train_loader_normal, train_loader_pair, test_loader, num_query, num_classes, num_cam, num_view = make_dataloader(cfg)
#         print("数据加载成功！")
#         print(f"num_query: {num_query}, num_classes: {num_classes}, num_cam: {num_cam}, num_view: {num_view}")
        
#         # 迭代一个批次验证格式
#         for n_iter, (img, pid, camid, viewid) in enumerate(train_loader):
#             print(f"训练集批次 {n_iter}：img.shape={img.shape}, pid.shape={pid.shape}, camid.shape={camid.shape}, viewid.shape={viewid.shape}")
#             break
        
#         for n_iter, (img, pid, camid, camids_batch, viewid) in enumerate(test_loader):
#             print(f"测试集批次 {n_iter}：img.shape={img.shape}, pid数量={len(pid)}, camid数量={len(camid)}")
#             break
#     except Exception as e:
#         print(f"加载失败：{e}")
#         raise