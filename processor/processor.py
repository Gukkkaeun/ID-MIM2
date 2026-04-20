import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.amp import autocast, GradScaler
import torch.distributed as dist



def pretrain(cfg, model, train_loader_pair, optimizer, scheduler, local_rank=0):
    """ID-MIM预训练主函数"""
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("ID-MIM.pretrain")
    logger.info("="*50 + " Start ID-MIM Pretraining " + "="*50)

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    scaler = GradScaler('cuda')


    # 开始预训练
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()

        scheduler.step(epoch)
        model.train()
        if hasattr(train_loader_pair, "sampler") and hasattr(train_loader_pair.sampler, "set_epoch"):
                train_loader_pair.sampler.set_epoch(epoch)

        for batch_idx, (img, pid, camid, viewid, img_wh) in enumerate(train_loader_pair):
            img = img.to(device)       # [B, 3, H, W] 输入图像
            camid = camid.to(device)       # [B] 模态标签（0=optical, 1=sar）
            pid = pid.cuda() if pid[0] != -1 else None  # 无标注时为None

            with autocast('cuda'):
                loss_dict = model(img, camid, img_wh, phase='pretrain')
                loss_total = loss_dict['loss_total']

            # 反向传播 + 优化器更新
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss_total.item(), img.shape[0])

            torch.cuda.synchronize()
            if (batch_idx + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                         epoch, (batch_idx + 1), len(train_loader_pair), loss_meter.avg, scheduler._get_lr(epoch)[0]
                    )
                )


            end_time = time.time()
            time_per_batch = (end_time - start_time) / (batch_idx + 1)

            if epoch % checkpoint_period == 0:
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))
                else:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))


    # 训练结束
    logger.info("="*50 + " ID-MIM Pretraining Finished " + "="*50)



def finetune(cfg, model, train_loader, optimizer, scheduler, local_rank=0):
    """ID-MIM微调主函数"""
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("ID-MIM.finetune")
    logger.info("="*50 + " Start ID-MIM Finetuning " + "="*50)

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    scaler = GradScaler('cuda')


    # 8. 开始微调
    logger.info("="*50 + " Start ID-MIM Finetuning " + "="*50)
    model.train()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()

        scheduler.step(epoch)
        model.train()
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

        for batch_idx, (img, pid, camid, viewid, mod) in enumerate(train_loader):
            img = img.to(device)       # [B, 3, H, W] 输入图像
            mod = mod.to(device)       # [B] 模态标签（0=optical, 1=sar）
            pid = pid.cuda() if pid[0] != -1 else None  # 无标注时为None

            with autocast('cuda'):
                loss_dict = model(img, mod, pid, phase='finetune')
                loss_total = loss_dict['loss_total']

            # 反向传播 + 优化器更新
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss_total.item(), img.shape[0])

            torch.cuda.synchronize()
            if (batch_idx + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                         epoch, (batch_idx + 1), len(train_loader), loss_meter.avg, scheduler._get_lr(epoch)[0]
                    )
                )


            end_time = time.time()
            time_per_batch = (end_time - start_time) / (batch_idx + 1)

            if epoch % checkpoint_period == 0:
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))
                else:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))


    # 训练结束
    logger.info("="*50 + " ID-MIM Finetuning Finished " + "="*50)



def test_idmim(cfg):
    """ID-MIM测试主函数"""
    # 1. 初始化日志
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("ID-MIM Test", cfg.OUTPUT_DIR, 0, if_train=False)
    logger.info(f"Using GPU: {cfg.MODEL.DEVICE_ID}")
    logger.info(f"Test Config: {cfg}")

    # 2. 构建测试数据加载器
    _, query_loader, gallery_loader, num_classes, num_cameras = make_dataloader(cfg, phase='finetune')
    logger.info(f"Test dataset loaded: # query samples: {len(query_loader.dataset)}, # gallery samples: {len(gallery_loader.dataset)}")

    # 3. 加载模型
    model = load_model(cfg, num_classes, num_cameras)

    # 4. 评估ReID核心指标
    logger.info("="*50 + " Start ReID Evaluation " + "="*50)
    rank1, rank5, rank10, mAP = evaluate_rank(model, query_loader, gallery_loader, cfg.TEST.FEAT_NORM == 'yes')
    logger.info(
        f"Final Evaluation Result | "
        f"Rank-1: {rank1:.2f}% | "
        f"Rank-5: {rank5:.2f}% | "
        f"Rank-10: {rank10:.2f}% | "
        f"mAP: {mAP:.2f}%"
    )

    # 5. 计算理论指标（特征-身份互信息）
    logger.info("="*30 + " Compute Theoretical Metrics " + "="*30)
    query_feat, query_pid, query_mod, _ = extract_features(model, query_loader)
    mi_score = compute_mutual_info_score(query_feat, query_pid)
    logger.info(f"Feature-Identity Mutual Information Score: {mi_score:.4f}")

    # 6. 可视化结果（TSNE + 注意力热力图 + 重建结果）
    logger.info("="*30 + " Generate Visualization Results " + "="*30)
    # 6.1 TSNE特征可视化（前500个样本）
    vis_num = min(500, len(query_feat))
    tsne_visualize(
        query_feat[:vis_num],
        query_pid[:vis_num],
        query_mod[:vis_num],
        save_path=os.path.join(cfg.OUTPUT_DIR, "tsne_feat.png")
    )
    logger.info(f"TSNE visualization saved to: {os.path.join(cfg.OUTPUT_DIR, 'tsne_feat.png')}")

    # 6.2 注意力热力图（随机选1个样本）
    random_idx = np.random.randint(0, len(query_loader.dataset))
    test_img, _, _, _, test_mod = query_loader.dataset[random_idx]
    test_img = test_img.unsqueeze(0).cuda()
    # 提取注意力权重（需修改骨干网络返回注意力）
    # attn_weights = model[0].get_attention(test_img, test_mod)
    # attention_heatmap(attn_weights, test_img.squeeze(0), save_path=os.path.join(cfg.OUTPUT_DIR, "attn_heatmap.png"))``

    # 6.3 重建结果可视化（预训练阶段）
    # recon_visualize(test_img.squeeze(0), recon_intra, recon_cross, save_path=os.path.join(cfg.OUTPUT_DIR, "recon_result.png"))

    logger.info("="*50 + " ID-MIM Test Finished " + "="*50)


def test(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("IDMIM.test")
    logger.info("Enter testing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for test".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for batch_idx, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]