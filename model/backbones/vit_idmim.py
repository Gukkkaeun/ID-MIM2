import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(-std*2, std*2)
        return tensor

# ===================== ID-MIM 特征解耦 =====================
class FeatureDisentangler(nn.Module):
    def __init__(self, embed_dim, num_heads=12):
        super().__init__()
        self.id_proj = nn.Linear(embed_dim, embed_dim)
        self.mod_proj = nn.Linear(embed_dim, embed_dim)
        self.noise_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        f_id = F.normalize(self.id_proj(x), dim=-1)
        f_mod = F.normalize(self.mod_proj(x), dim=-1)
        f_noise = F.normalize(self.noise_proj(x), dim=-1)
        return f_id, f_mod, f_noise

# ====================== 修复版：可直接运行 =====================
class ViT_IDMIM(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        camera=3,  # VIS/NIR/TIR = 3个模态
        mlp_ratio=4.,
        qkv_bias=True
    ):
        super().__init__()
        # 双模态编码（光学 / SAR 或 NIR/TIR）
        self.patch_embed_rgb = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.patch_embed_sar = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed_rgb.num_patches

        # CLS token + 位置编码 + 模态编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))

        # 图像宽高嵌入 SSE
        self.wh_embed = nn.Linear(2, embed_dim)

        self.embed_dim = embed_dim

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 解耦头
        self.disentangler = FeatureDisentangler(embed_dim, num_heads)

        # 初始化
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.mie_embed, std=.02)

    def forward(self, x, cam_id, img_wh):
        B = x.shape[0]

        # ---------------------- 双模态编码 ----------------------
        rgb_idx = cam_id == 0
        sar_idx = cam_id != 0  
        feat = torch.zeros(
            B, self.patch_embed_rgb.num_patches, 768, 
            dtype=x.dtype, device=x.device
        )

        if rgb_idx.any():
            feat[rgb_idx] = self.patch_embed_rgb(x[rgb_idx])
        if sar_idx.any():
            feat[sar_idx] = self.patch_embed_sar(x[sar_idx])

        # ---------------------- CLS + PE + MIE ----------------------
        cls = self.cls_token.expand(B, -1, -1)
        feat = torch.cat((cls, feat), dim=1)
        feat = feat + self.pos_embed + self.mie_embed[cam_id]

        # ---------------------- 宽高嵌入 SSE ----------------------
        wh = self.wh_embed(img_wh).unsqueeze(1)  # [B, 1, 768]
        feat = torch.cat((feat, wh), dim=1)

        # ---------------------- Transformer ----------------------
        for blk in self.blocks:
            feat = blk(feat)

        feat = self.norm(feat)

        # ---------------------- 特征解耦 ----------------------
        f_id, f_mod, f_noise = self.disentangler(feat)

        # 只返回 CLS 特征
        return f_id[:, 0], f_mod[:, 0], f_noise[:, 0]


def vit_base_patch16_224_IDMIM(**kwargs):
    model = ViT_IDMIM(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        **kwargs
    )
    return model


# if __name__ == "__main__":
#     model = vit_base_patch16_224_IDMIM(camera=3)
    
#     # 模拟输入
#     x = torch.randn(4, 3, 224, 224)       # 图片
#     cam_id = torch.tensor([0,1,2,0])      # 模态
#     img_wh = torch.randn(4, 2)            # 宽高
    
#     f_id, f_mod, f_noise = model(x, cam_id, img_wh)
#     print("f_id shape:", f_id.shape)      # torch.Size([4, 768])
#     print("模型运行成功 ✅")