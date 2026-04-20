"""
构建ID-MIM模型（预训练/微调）
"""
import torch
import torch.nn as nn
from .backbones.vit_idmim import vit_base_patch16_224_IDMIM
from .backbones.vit_transoss import vit_base_patch16_224_TransOSS
from .id_mim import IDMIM
from loss.metric_learning import Arcface


def make_model(cfg, num_class=0, num_cam=0, phase='pretrain'):
    """
    构建ID-MIM模型
    Args:
        cfg: 配置对象
        num_class: 身份类别数（微调时用）
        num_cam: 相机/模态数量（微调时用）
        phase: 'pretrain'/'finetune'
    Returns:
        model: ID-MIM模型
    """
    if phase == 'pretrain':
        # 预训练：使用ID-MIM ViT骨干
        backbone = vit_base_patch16_224_IDMIM(camera=num_cam)  # 传入 camera 数
        model = IDMIM(cfg, backbone)

    elif phase == 'finetune':
        # 微调：使用TransOSS骨干（加载ID-MIM预训练权重）
        backbone = vit_base_patch16_224_TransOSS(
            img_size=cfg.INPUT.SIZE_TRAIN,
            camera=num_cam,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            mie_coe=cfg.MODEL.MIE_COE,
            sse=cfg.MODEL.SSE
        )

        # 加载ID-MIM预训练权重
        if cfg.MODEL.PRETRAIN_CHOICE == 'id_mim':
            print(f"==================== Loading ID-MIM pretrain weights ====================")
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cpu')
            
            # 兼容多种 checkpoint 格式
            if 'state_dict' in checkpoint:
                pretrain_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                pretrain_dict = checkpoint['model']
            else:
                pretrain_dict = checkpoint  # 直接是权重

            # 只保留 backbone 中存在的 key（自动过滤不匹配层）
            backbone_dict = backbone.state_dict()
            matched_dict = {}
            for k, v in pretrain_dict.items():
                # 去掉可能的前缀 "backbone."
                k = k.replace("backbone.", "")
                if k in backbone_dict and v.shape == backbone_dict[k].shape:
                    matched_dict[k] = v

            # 加载匹配的权重
            backbone.load_state_dict(matched_dict, strict=False)
            print(f"✅ Successfully loaded {len(matched_dict)} matched layers")

        # ===================== 微调模型结构 =====================
        class FinetuneModel(nn.Module):
            def __init__(self, backbone, num_classes, s=30.0, m=0.3):
                super().__init__()
                self.backbone = backbone
                self.bn = nn.BatchNorm1d(768)
                self.arcface = Arcface(768, num_classes, s=s, m=m)

            def forward(self, x, cam_label=None, img_wh=None, label=None):
                # 训练：返回 arcface logits
                if label is not None:
                    feat = self.backbone(x, cam_label, cam_label)
                    feat = self.bn(feat)
                    return self.arcface(feat, label)
                # 测试：只返回特征
                else:
                    feat = self.backbone(x, cam_label, img_wh)
                    return self.bn(feat)

        model = FinetuneModel(
            backbone,
            num_class,
            s=cfg.SOLVER.COSINE_SCALE,
            m=cfg.SOLVER.COSINE_MARGIN
        )

    else:
        raise ValueError(f"Unsupported phase: {phase}")

    return model