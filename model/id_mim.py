"""
ID-MIM核心模块：身份感知掩码+跨模态重建+身份一致性正则化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IdentityAwareMasking(nn.Module):
    """身份感知最优掩码策略"""
    def __init__(self, mask_ratio_id=0.15, mask_ratio_non_id=0.8, embed_dim=768):
        super().__init__()
        self.mask_ratio_id = mask_ratio_id      # 身份区域掩码率
        self.mask_ratio_non_id = mask_ratio_non_id # 非身份区域掩码率
        self.embed_dim = embed_dim
        # 轻量级特征提取器（计算patch与身份的互信息）
        self.mi_estimator = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def compute_id_relevance(self, x, mod):
        """
        计算每个patch的身份相关性（互信息）
        Args:
            x: [B, N, C] patch特征
            mod: [B] 模态标签（0:optical, 1:sar）
        Returns:
            mi_scores: [B, N] 每个patch的身份互信息得分
        """
        B, N, C = x.shape
        # 展平特征计算互信息
        x_flat = x.reshape(-1, C)  # [B*N, C]
        mi_scores = self.mi_estimator(x_flat).reshape(B, N)  # [B, N]

        # 模态自适应调整
        # 光学图像：增强纹理区域的得分
        opt_mask = (mod == 0).unsqueeze(1).expand(B, N)
        # SAR图像：增强后向散射区域的得分
        sar_mask = (mod == 1).unsqueeze(1).expand(B, N)

        # 光学图像：基于纹理熵调整
        x_var = x.var(dim=-1)  # [B, N] 纹理熵
        mi_scores[opt_mask] = mi_scores[opt_mask] * (1 + x_var[opt_mask])
        # SAR图像：基于强度调整
        x_mean = x.mean(dim=-1) # [B, N] 散射强度
        mi_scores[sar_mask] = mi_scores[sar_mask] * (1 + x_mean[sar_mask])

        return mi_scores

    def forward(self, x, mod):
        """
        生成身份感知掩码
        Args:
            x: [B, N, C] patch特征
            mod: [B] 模态标签
        Returns:
            mask: [B, N] 掩码（1=掩码，0=保留）
            ids_patches: [B, N] 身份相关patch掩码（1=身份区域）
        """
        B, N, C = x.shape
        # 计算身份相关性得分
        mi_scores = self.compute_id_relevance(x, mod)  # [B, N]

        # 分位数划分身份/非身份区域（自适应阈值）
        threshold = torch.quantile(mi_scores, 0.7, dim=1, keepdim=True)  # [B, 1]
        ids_patches = (mi_scores >= threshold).float()  # [B, N] 身份区域
        non_ids_patches = (mi_scores < threshold).float() # [B, N] 非身份区域

        # 生成掩码
        mask = torch.zeros_like(mi_scores)
        # 身份区域低掩码率
        mask[ids_patches == 1] = torch.bernoulli(
            torch.ones_like(mask[ids_patches == 1]) * self.mask_ratio_id)
        # 非身份区域高掩码率
        mask[non_ids_patches == 1] = torch.bernoulli(
            torch.ones_like(mask[non_ids_patches == 1]) * self.mask_ratio_non_id)

        return mask.bool(), ids_patches

class CrossModalReconstructor(nn.Module):
    """身份解耦跨模态双向重建模块"""
    def __init__(self, embed_dim=768, patch_size=16, img_size=224):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) **2

        # 模态内重建头
        self.intra_recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 3 * patch_size * patch_size)
        )

        # 跨模态重建头（仅用身份特征）
        self.cross_recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 3 * patch_size * patch_size)
        )

    def unfold_img(self, img):
        """将图像展开为patches"""
        B, C, H, W = img.shape
        patches = F.unfold(
            img, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.permute(0, 2, 1).reshape(
            B, self.num_patches, C, self.patch_size, self.patch_size)
        return patches

    def fold_patches(self, patches):
        """将patches折叠为图像"""
        B, N, C, PH, PW = patches.shape
        patches = patches.reshape(B, N, C * PH * PW).permute(0, 2, 1)
        img = F.fold(
            patches, output_size=(self.img_size, self.img_size),
            kernel_size=self.patch_size, stride=self.patch_size)
        return img

    def forward(self, f_id, f_mod, img, mod, mask):
        """
        跨模态重建
        Args:
            f_id: [B, N+1, C] 身份特征（含CLS）
            f_mod: [B, N+1, C] 模态特征（含CLS）
            img: [B, 3, H, W] 原始图像
            mod: [B] 模态标签
            mask: [B, N+1] 掩码（True=掩码）
        Returns:
            recon_intra: [B, 3, H, W] 模态内重建图像
            recon_cross: [B, 3, H, W] 跨模态重建图像
        """
        B = img.shape[0]
        # 移除CLS token
        f_id = f_id[:, 1:, :]  # [B, N, C]
        f_mod = f_mod[:, 1:, :] # [B, N, C]

        # 掩码特征（仅保留未掩码部分）
        f_id_masked = f_id[~mask[:, 1:]]  # [B*N_unmasked, C]
        f_mod_masked = f_mod[~mask[:, 1:]] # [B*N_unmasked, C]

        # 模态内重建：f_id + f_mod
        f_intra = f_id_masked + f_mod_masked
        patches_intra = self.intra_recon_head(f_intra).reshape(
            -1, 3, self.patch_size, self.patch_size)
        # 恢复掩码位置
        recon_intra_patches = torch.zeros(
            B, self.num_patches, 3, self.patch_size, self.patch_size,
            device=patches_intra.device)
        recon_intra_patches[~mask[:, 1:]] = patches_intra
        recon_intra = self.fold_patches(recon_intra_patches)

        # 跨模态重建：仅用f_id
        f_cross = f_id_masked
        patches_cross = self.cross_recon_head(f_cross).reshape(
            -1, 3, self.patch_size, self.patch_size)
        recon_cross_patches = torch.zeros(
            B, self.num_patches, 3, self.patch_size, self.patch_size,
            device=patches_cross.device)
        recon_cross_patches[~mask[:, 1:]] = patches_cross
        recon_cross = self.fold_patches(recon_cross_patches)

        return recon_intra, recon_cross

class IdentityConsistencyRegularizer(nn.Module):
    """身份一致性正则化模块"""
    def __init__(self, tau=0.05):
        super().__init__()
        self.tau = tau # 温度系数

    def forward(self, f_id_opt, f_id_sar, pid):
        """
        计算身份一致性损失
        Args:
            f_id_opt: [B_opt, C] 光学图像身份特征
            f_id_sar: [B_sar, C] SAR图像身份特征
            pid: [B] 身份标签（配对的光学/SAR）
        Returns:
            loss_id_con: 标量 身份一致性损失
        """
        # 归一化特征
        f_id_opt = F.normalize(f_id_opt, dim=-1)
        f_id_sar = F.normalize(f_id_sar, dim=-1)

        # 计算余弦相似度
        sim_matrix = torch.matmul(f_id_opt, f_id_sar.t()) / self.tau # [B_opt, B_sar]

        # InfoNCE损失
        labels = torch.arange(len(pid), device=sim_matrix.device)
        loss_id_con = F.cross_entropy(sim_matrix, labels)

        # 掩码不变性约束（简化版）
        loss_mask_inv = torch.mean((f_id_opt.var(dim=0) - f_id_sar.var(dim=0))** 2)

        return loss_id_con + 0.3 * loss_mask_inv

class IDMIM(nn.Module):
    """ID-MIM整体框架"""
    def __init__(self, cfg, backbone):
        super().__init__()
        self.backbone = backbone # ID-MIM ViT骨干
        self.embed_dim = backbone.embed_dim

        # 身份感知掩码模块
        self.id_masking = IdentityAwareMasking(
            mask_ratio_id = cfg.MODEL.ID_MIM.MASK_RATIO_ID,
            mask_ratio_non_id = cfg.MODEL.ID_MIM.MASK_RATIO_NON_ID,
            embed_dim = self.embed_dim
        )

        # 跨模态重建模块
        self.cross_modal_recon = CrossModalReconstructor(
            embed_dim = self.embed_dim,
            patch_size = 16,
            img_size = cfg.INPUT.SIZE_TRAIN[0]
        )

        # 身份一致性正则化模块
        self.id_consistency = IdentityConsistencyRegularizer(
            tau = cfg.MODEL.ID_MIM.TAU_ID_CON
        )

        # 损失权重
        self.lambda_cross = cfg.MODEL.ID_MIM.LAMBDA_CROSS
        self.beta_orth = cfg.MODEL.ID_MIM.BETA_ORTH
        self.alpha_reg = cfg.MODEL.ID_MIM.ALPHA_REG

    def compute_orthogonal_loss(self, f_id, f_mod, f_noise):
        """计算正交约束损失"""
        # 计算特征间的内积
        orth_id_mod = torch.mean((f_id.transpose(1, 2) @ f_mod) **2)
        orth_id_noise = torch.mean((f_id.transpose(1, 2) @ f_noise)** 2)
        return orth_id_mod + orth_id_noise

    def forward_pretrain(self, img, mod, pid=None):
        """
        预训练前向传播
        Args:
            img: [B, 3, H, W] 输入图像
            mod: [B] 模态标签
            pid: [B] 身份标签（可选）
        Returns:
            loss_dict: 损失字典
        """
        # 1. 骨干网络提取并解耦特征
        f_id, f_mod, f_noise = self.backbone(img) # [B, N+1, C]

        # 2. 生成身份感知掩码
        mask, ids_patches = self.id_masking(f_id, mod)

        # 3. 跨模态重建
        recon_intra, recon_cross = self.cross_modal_recon(
            f_id, f_mod, img, mod, mask
        )

        # 4. 计算损失
        # 重建损失
        loss_intra = F.l1_loss(recon_intra, img)
        loss_cross = F.l1_loss(recon_cross, img)
        loss_recon = loss_intra + self.lambda_cross * loss_cross

        # 正交约束损失
        loss_orth = self.compute_orthogonal_loss(f_id, f_mod, f_noise)

        # 身份一致性损失（仅当有pid时）
        loss_id_con = torch.tensor(0.0, device=img.device)
        if pid is not None and len(pid) > 0:
            # 分离光学/SAR特征
            opt_mask = (mod == 0)
            sar_mask = (mod == 1)
            if opt_mask.sum() > 0 and sar_mask.sum() > 0:
                f_id_opt = f_id[opt_mask][:, 0, :] # CLS token作为全局身份特征
                f_id_sar = f_id[sar_mask][:, 0, :]
                loss_id_con = self.id_consistency(f_id_opt, f_id_sar, pid)

        # 总损失
        loss_total = loss_recon + self.beta_orth * loss_orth + self.alpha_reg * loss_id_con

        loss_dict = {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_orth': loss_orth,
            'loss_id_con': loss_id_con
        }

        return loss_dict

    def forward_finetune(self, img, mod):
        """
        微调前向传播（提取身份特征）
        Args:
            img: [B, 3, H, W] 输入图像
            mod: [B] 模态标签
        Returns:
            f_id_global: [B, C] 全局身份特征
        """
        # 提取解耦特征
        f_id, _, _ = self.backbone(img)
        # 用CLS token作为全局身份特征
        f_id_global = f_id[:, 0, :]
        return f_id_global

    def forward(self, img, mod, pid = None, phase ='pretrain'):
        """
        统一前向传播
        Args:
            img: [B, 3, H, W] 输入图像
            mod: [B] 模态标签
            pid: [B] 身份标签
            phase: 'pretrain'/'finetune'
        Returns:
            预训练：loss_dict | 微调：f_id_global
        """
        if phase == 'pretrain':
            return self.forward_pretrain(img, mod, pid)
        elif phase == 'finetune':
            return self.forward_finetune(img, mod)
        else:
            raise ValueError(f"Unsupported phase: {phase}")