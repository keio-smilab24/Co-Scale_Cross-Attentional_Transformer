import torch
import torch.nn as nn
from torch import einsum
import init
from sscdnet.correlation_package.correlation import Correlation

from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

from transformer import MultiHeadAttention


class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dim=512, feedforward_dim=1024, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim

        self.fc1 = nn.Linear(self.model_dim, self.feedforward_dim)
        self.gelu = nn.GELU()
        self.dp = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.feedforward_dim, self.model_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = self.dp(x)
        return x


class PatchEmbed(nn.Module):
    """
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16, stride=16, padding=0):
        super().__init__()
        self.stride = stride
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        out_h, out_w = H // self.stride, W // self.stride

        x = self.proj(x).flatten(2).transpose(1, 2)
        out = self.norm(x)

        return out, (out_h, out_w)


class FactorAttnConvRelPosEnc(nn.Module):
    """
    Modified from timm/models/coat.py#L90
    """
    def __init__(self, model_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1, shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = model_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(model_dim, model_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, B, h, N, Ch]
        q, k, v = qkv[0], qkv[1], qkv[2]                                                 # [B, h, N, Ch]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)                                                     # Softmax on dim N
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)          # [B, h, Ch, Ch]
        factor_att        = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # [B, h, N, Ch]

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)                                                # [B, h, N, Ch]

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)                                           # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvRelPosEnc(nn.Module):
    """
    Modified from timm/models/coat.py#L24
    """
    def __init__(self, Ch, h, window):
        super().__init__()

        if isinstance(window, int):
            window = {window: h}                                                         # Set the same window size for all attention heads.
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()            
        
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = nn.Conv2d(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=(cur_window, cur_window), 
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),                          
                groups=cur_head_split*Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]
    
    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        if N == (1 + H * W):
            q_img = q[:,:,1:,:]
            v_img = v[:,:,1:,:]
        
            v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)
            v_img_list = torch.split(v_img, self.channel_splits, dim=1)
            conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
            conv_v_img = torch.cat(conv_v_img_list, dim=1)
            conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)

            EV_hat_img = q_img * conv_v_img
            zero = torch.zeros((B, h, 1, Ch), dtype=q.dtype, layout=q.layout, device=q.device)
            EV_hat = torch.cat((zero, EV_hat_img), dim=2)
        else:
            q_img = q
            v_img = v
        
            v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)
            v_img_list = torch.split(v_img, self.channel_splits, dim=1)
            conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
            conv_v_img = torch.cat(conv_v_img_list, dim=1)
            conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)

            EV_hat_img = q_img * conv_v_img
            EV_hat = EV_hat_img

        return EV_hat


class ConvPosEnc(nn.Module):
    """
    Modified from timm/models/coat.py#L140
    """
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim) 
    
    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        if N == (1 + H * W):
            # Extract CLS token and image tokens.
            cls_token, img_tokens = x[:, :1], x[:, 1:]                                       # Shape: [B, 1, C], [B, H*W, C].
            
            # Depthwise convolution.
            feat = img_tokens.transpose(1, 2).view(B, C, H, W)
            x = self.proj(feat) + feat
            x = x.flatten(2).transpose(1, 2)

            # Combine with CLS token.
            x = torch.cat((cls_token, x), dim=1)
        else:
            img_tokens = x
            feat = img_tokens.transpose(1, 2).view(B, C, H, W)
            x = self.proj(feat) + feat
            x = x.flatten(2).transpose(1, 2)

        return x


class SerialBlock_one(nn.Module):
    """
    Modified from timm/models/coat.py#L167
    """
    def __init__(self, model_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, shared_cpe=None, shared_crpe=None):
        super().__init__()

        # Conv-Attention
        self.cpe = shared_cpe

        self.norm1 = nn.LayerNorm(model_dim, eps=1e-6)
        self.factoratt_crpe = FactorAttnConvRelPosEnc(model_dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # FeedForwardNetwork
        self.norm2 = nn.LayerNorm(model_dim, eps=1e-6)
        hidden_dim = int(model_dim * mlp_ratio)
        self.ffn = FeedForwardNetwork(model_dim, hidden_dim, drop)
    
    def forward(self, x, size):
        # Conv-Attention
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)

        cur = self.norm2(x)
        cur = self.ffn(cur)
        x = x + self.drop_path(cur)

        return x


class SerialBlock_full(nn.Module):
    """
    Modified from timm/models/coat.py#L353
    """
    def __init__(self, patch_size=16, stride=16, in_chans=3, embed_dims=[0, 0, 0, 0], 
                 serial_depths=[0, 0, 0, 0], num_heads=8, mlp_ratios=[0, 0, 0, 0], 
                 qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0, crpe_window={3:2, 5:3, 7:3}):
        super().__init__()

        # Patch embeddings.
        self.patch_embed1 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0], patch_size=patch_size, stride=stride)
        self.patch_embed2 = PatchEmbed(in_chans=embed_dims[0], embed_dim=embed_dims[1], patch_size=2, stride=2)
        self.patch_embed3 = PatchEmbed(in_chans=embed_dims[1], embed_dim=embed_dims[2], patch_size=2, stride=2)
        self.patch_embed4 = PatchEmbed(in_chans=embed_dims[2], embed_dim=embed_dims[3], patch_size=2, stride=2)

        # Class tokens.
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # Convolutional position encodings.
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)

        # Convolutional relative position encodings.
        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)

        # Enable stochastic depth.
        dpr = drop_path_rate
        
        # Serial blocks 1.
        self.serial_blocks1 = nn.ModuleList([
            SerialBlock_one(
                embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe1, shared_crpe=self.crpe1
            )
            for _ in range(serial_depths[0])]
        )

        # Serial blocks 2.
        self.serial_blocks2 = nn.ModuleList([
            SerialBlock_one(
                embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe2, shared_crpe=self.crpe2
            )
            for _ in range(serial_depths[1])]
        )

        # Serial blocks 3.
        self.serial_blocks3 = nn.ModuleList([
            SerialBlock_one(
                embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, 
                shared_cpe=self.cpe3, shared_crpe=self.crpe3
            )
            for _ in range(serial_depths[2])]
        )

        # Serial blocks 4.
        self.serial_blocks4 = nn.ModuleList([
            SerialBlock_one(
                embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, 
                shared_cpe=self.cpe4, shared_crpe=self.crpe4
            )
            for _ in range(serial_depths[3])]
        )

        # Initialize weights.
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.cls_token3, std=.02)
        trunc_normal_(self.cls_token4, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4'}
    
    def insert_cls(self, x, cls_token):
        """ Insert CLS token. """
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def remove_cls(self, x):
        """ Remove CLS token. """
        return x[:, 1:, :]
    
    def forward(self, x0):
        B = x0.shape[0]

        # Serial blocks 1.
        x1, (H1, W1) = self.patch_embed1(x0)
        x1 = self.insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = self.remove_cls(x1)
        x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        
        # Serial blocks 2.
        x2, (H2, W2) = self.patch_embed2(x1_nocls)
        x2 = self.insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = self.remove_cls(x2)
        x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 3.
        x3, (H3, W3) = self.patch_embed3(x2_nocls)
        x3 = self.insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = self.remove_cls(x3)
        x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 4.
        x4, (H4, W4) = self.patch_embed4(x3_nocls)
        x4 = self.insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = self.remove_cls(x4)
        x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
    
        return (x1_nocls, H1, W1), (x2_nocls, H2, W2), (x3_nocls, H3, W3), (x4_nocls, H4, W4)


class DecoderBlock(nn.Module):
    """
    Modified from timm/models/coat.py#L167
    """
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1):
        super().__init__()

        # Conv-Attention
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window={3:2, 5:3, 7:3})
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.factoratt_crpe = FactorAttnConvRelPosEnc(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, self.crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # FeedForwardNetwork
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = FeedForwardNetwork(model_dim=dim, feedforward_dim=hidden_dim, dropout=drop)

    def forward(self, x0, size):
        B = x0.shape[0]
        (H, W) = size

        x0 = self.cpe(x0, size=(H, W))
        cur0 = self.norm1(x0)
        cur0 = self.factoratt_crpe(cur0, size=(H,W))
        x0 = x0 + self.drop_path(cur0)

        cur = self.norm2(x0)
        cur = self.ffn(cur)
        x = x0 + self.drop_path(cur)

        return x


class Cscat(nn.Module):
    def __init__(self, patch_size=16, stride=16, in_chans=3, embed_dims=[0, 0, 0, 0], 
                 serial_depths=[0, 0, 0, 0], num_heads=8, mlp_ratios=[0, 0, 0, 0], 
                 qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0, crpe_window={3:2, 5:3, 7:3}):
        super().__init__()

        self.enc1_serial = SerialBlock_full(patch_size, stride, in_chans, embed_dims, 
                 serial_depths, num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                 drop_path_rate, crpe_window)
                
        self.enc2_serial = SerialBlock_full(patch_size, stride, in_chans, embed_dims, 
                 serial_depths, num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                 drop_path_rate, crpe_window)
        
        self.patch_embed4 = PatchEmbed(embed_dims[3]*3+embed_dims[2]+21*21, embed_dim=embed_dims[2], patch_size=1, stride=1)
        self.patch_embed3 = PatchEmbed(in_chans=embed_dims[2]*4+embed_dims[1], embed_dim=embed_dims[1], patch_size=1, stride=1)
        self.patch_embed2 = PatchEmbed(in_chans=embed_dims[1]*3+embed_dims[0]+21*21, embed_dim=embed_dims[0], patch_size=1, stride=1)
        self.patch_embed1 = PatchEmbed(in_chans=embed_dims[0]*3, embed_dim=embed_dims[0] // 2, patch_size=1, stride=1)
        self.patch_embed0 = PatchEmbed(32, 16, patch_size=1, stride=1)
        self.patch_embed00 = PatchEmbed(16, 8, patch_size=1, stride=1)
        
        self.decoder4 = DecoderBlock(embed_dims[2], num_heads, mlp_ratios[3], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, drop_path_rate)
        self.decoder3 = DecoderBlock(embed_dims[1], num_heads, mlp_ratios[2], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, drop_path_rate)
        self.decoder2 = DecoderBlock(embed_dims[0], num_heads, mlp_ratios[1], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, drop_path_rate)
        self.decoder1 = DecoderBlock(embed_dims[0] // 2, num_heads, mlp_ratios[0], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, drop_path_rate)
        self.decoder0 = DecoderBlock(16, num_heads, 4, qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, drop_path_rate)
        self.decoder00 = DecoderBlock(8, num_heads, 4, qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, drop_path_rate)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1_1 = nn.Conv2d(embed_dims[0]*2, embed_dims[0], 1)
        self.down1_2 = nn.Conv2d(embed_dims[0], embed_dims[0], 2, 2)
        self.down2_1 = nn.Conv2d(embed_dims[1]*2+embed_dims[0], embed_dims[1], 1)
        self.down2_2 = nn.Conv2d(embed_dims[1], embed_dims[1], 2, 2)
        self.down3_1 = nn.Conv2d(embed_dims[2]*3+embed_dims[1], embed_dims[2], 1)
        self.down3_2 = nn.Conv2d(embed_dims[2], embed_dims[2], 2, 2)
        self.down4 = nn.Conv2d(embed_dims[3]*2+embed_dims[2], embed_dims[3], 1)

        self.tf2 = MultiHeadAttention(embed_dims[1])
        self.tf3 = MultiHeadAttention(embed_dims[2])
        self.tf4 = MultiHeadAttention(embed_dims[3])
        self.dec_tf = MultiHeadAttention(embed_dims[2])

        self.dec_corr4 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.dec_corr3 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.dec_corr2 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.dec_corr1 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, enc1, enc2):
        B = enc1.shape[0]
        (enc1_x1_nocls, H1, W1), (enc1_x2_nocls, H2, W2), (enc1_x3_nocls, H3, W3), (enc1_x4_nocls, H4, W4) = self.enc1_serial(enc1)
        (enc2_x1_nocls, H1, W1), (enc2_x2_nocls, H2, W2), (enc2_x3_nocls, H3, W3), (enc2_x4_nocls, H4, W4) = self.enc2_serial(enc2)

        cor4 = self.dec_corr4(enc1_x4_nocls, enc2_x4_nocls)
        cor4 = self.corr_activation(cor4)
        cor3 = self.dec_corr3(enc1_x3_nocls, enc2_x3_nocls)
        cor3 = self.corr_activation(cor3)
        cor2 = self.dec_corr2(enc1_x2_nocls, enc2_x2_nocls)
        cor2 = self.corr_activation(cor2)
        cor1 = self.dec_corr1(enc1_x1_nocls, enc2_x1_nocls)
        cor1 = self.corr_activation(cor1)

        x1 = torch.cat([enc1_x1_nocls, enc2_x1_nocls], 1)
        x2 = self.down1_1(x1)
        x2 = self.down1_2(x2)
        x2 = torch.cat([x2, enc1_x2_nocls, enc2_x2_nocls], 1)
        x3 = self.down2_1(x2)
        x3 = self.down2_2(x3)
        x3_ = self.tf3(enc1_x3_nocls, enc2_x3_nocls, enc1_x3_nocls)
        x3 = torch.cat([x3, x3_, enc1_x3_nocls, enc2_x3_nocls], 1)
        x4 = self.down3_1(x3)
        x4 = self.down3_2(x4)
        x4 = torch.cat([x4, enc1_x4_nocls, enc2_x4_nocls], 1)
        dec = self.down4(x4)

        dec = torch.cat([dec, x4, cor4], 1)
        dec, _ = self.patch_embed4(dec)
        dec = self.decoder4(dec, (H4, W4))
        dec = dec.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        dec = self.up4(dec)
        dec = torch.cat([dec, x3], 1)
        dec, _ = self.patch_embed3(dec)
        dec = self.decoder3(dec, (H3, W3))
        dec = dec.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        dec = self.up3(dec)
        dec = torch.cat([dec, x2, cor2], 1)
        dec, _ = self.patch_embed2(dec)
        dec = self.decoder2(dec, (H2, W2))
        dec = dec.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        dec = self.up2(dec)
        dec = torch.cat([dec, x1], 1)
        dec, _ = self.patch_embed1(dec)
        dec = self.decoder1(dec, (H1, W1))
        dec = dec.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        dec = self.up1(dec)

        return dec


class Model(nn.Module):
    def __init__(self, inc, outc):
        super(Model, self).__init__()

        # encoder1
        self.enc1_conv1 = nn.Conv2d(int(inc/2), 64, 7, padding=3, stride=2, bias=False)
        self.enc1_bn1   = nn.BatchNorm2d(64)
        self.enc1_pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # encoder2
        self.enc2_conv1 = nn.Conv2d(int(inc/2), 64, 7, padding=3, stride=2, bias=False)
        self.enc2_bn1   = nn.BatchNorm2d(64)
        self.enc2_pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.coat = Cscat(4, 4, 3, [64, 128, 320, 512], [2, 2, 2, 2], num_heads=8, mlp_ratios=[8, 8, 4, 4])

        self.dec_bn1   = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, outc, 1, padding=0, stride=1)

        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)
        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_relu(self.modules())

    def forward(self, x):
        x1, x2 = torch.split(x,3,1)
        dec = self.coat(x1, x2)
        dec = self.dec_bn1(dec)
        dec = self.relu(dec)

        out = self.classifier(dec)

        return out
