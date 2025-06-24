import torch
import torchvision as tv
import torch.nn as nn
from timm.models.swin_transformer_v2 import SwinTransformerV2, SwinTransformerBlock, PatchMerging, _assert, window_partition, window_reverse, to_2tuple
from functools import reduce
from operator import mul
from torch.nn import Dropout, Conv2d
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
    dim (int): Number of input channels.
    window_size (tuple[int]): The height and width of the window.
    num_heads (int): Number of attention heads.
    qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """
    def __init__(self, num_prompts, prompt_location, dim, window_size, num_heads, 
                qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                pretrained_window_size=None):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([
            relative_coords_h,
            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x, mask = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        # 프롬프트 위치 적용
        if self.prompt_location == "prepend":
            _C, _H, _W = relative_position_bias.shape
            relative_position_bias = torch.cat((
                torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                relative_position_bias
            ), dim=1)
            relative_position_bias = torch.cat((
                torch.zeros(_C, _H + self.num_prompts, self.num_prompts, device=attn.device),
                relative_position_bias
            ), dim=-1)
        
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            if self.prompt_location == "prepend":
                mask = torch.cat((
                    torch.zeros(nW, self.num_prompts, _W, device=attn.device),
                    mask
                ), dim=1)
                mask = torch.cat((
                    torch.zeros(nW, _H + self.num_prompts, self.num_prompts, device=attn.device),
                    mask
                ), dim=-1)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PromptedSwinTransformerBlock(SwinTransformerBlock):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pretraining.
    """

    def __init__(
            self,  num_prompts, prompt_location, dim, input_resolution, num_heads, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__(dim, input_resolution, num_heads, window_size, shift_size,
                        mlp_ratio, qkv_bias, drop, attn_drop, drop_path,
                        act_layer, norm_layer, pretrained_window_size)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.prompt_location == "prepend":
            self.attn = WindowAttention(
                num_prompts, prompt_location,
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                pretrained_window_size=to_2tuple(pretrained_window_size))
    # def _attn(self, x, global_prompt=None):
    #     H, W = self.input_resolution
    #     B, L, C = x.shape
    #     _assert(L == H * W, "input feature has wrong size")
    #     x = x.view(B, H, W, C)

    #     # cyclic shift
    #     has_shift = any(self.shift_size)
    #     if has_shift:
    #         shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    #     else:
    #         shifted_x = x

    #     # partition windows
    #     x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    #     x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

    #     # QKV 분리
    #     B_, N, C = x_windows.shape
    #     qkv = self.attn.qkv(x_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
    #     qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        
    #     q, k, v = qkv[0], qkv[1], qkv[2]  # Query, Key, Value 분리

    #     if global_prompt is not None:
    #         # Prompt를 Query(Q)에 추가
    #         global_prompt = global_prompt.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, num_heads, num_prompts, head_dim]
    #         q = torch.cat([global_prompt, q], dim=2)  # [batch, num_heads, num_prompts + seq_len, head_dim]

    #     # Attention 연산 수행
    #     attn_weights = (q @ k.transpose(-2, -1)) * self.attn.scale
    #     attn_weights = attn_weights.softmax(dim=-1)
    #     attn_weights = self.attn.attn_drop(attn_weights)

    #     out = (attn_weights @ v).transpose(1, 2).reshape(B_, N, C)

    #     out = self.attn.proj(out)
    #     out = self.attn.proj_drop(out)

    #     # merge windows
    #     attn_windows = out.view(-1, self.window_size[0], self.window_size[1], C)
    #     shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)  # B H' W' C

    #     # reverse cyclic shift
    #     if has_shift:
    #         x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    #     else:
    #         x = shifted_x
    #     x = x.view(B, H * W, C)
    #     return x
    def _attn(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts
        _assert(L == H * W, "input feature has wrong size")
        x = x.view(B, H, W, C)

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        num_windows = int(x_windows.shape[0] / B)
        if self.prompt_location == "prepend":
            # expand prompts_embs
            # B, num_prompts, C --> nW*B, num_prompts, C
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
            x_windows = torch.cat((prompt_emb, x_windows), dim=1)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = attn_windows[:, :self.num_prompts, :]
            attn_windows = attn_windows[:, self.num_prompts:, :]
            # change prompt_embs's shape:
            # nW*B, num_prompts, C - B, num_prompts, C
            prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
            prompt_emb = prompt_emb.mean(0)
            
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)  # B H' W' C

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)
        return x
class PromptedPatchMerging(PatchMerging):
    r""" Patch Merging Layer.
    
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self, num_prompts, prompt_location, deep_prompt, input_resolution,
        dim, norm_layer=nn.LayerNorm
    ):
        super(PromptedPatchMerging, self).__init__(
            input_resolution, dim, norm_layer)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if prompt_location == "prepend":
            if not deep_prompt:
                self.prompt_upsampling = None
                # self.prompt_upsampling = nn.Linear(dim, 4 * dim, bias=False)
            else:
                self.prompt_upsampling = None

    def upsample_prompt(self, prompt_emb):
        if self.prompt_upsampling is not None:
            prompt_emb = self.prompt_upsampling(prompt_emb)
        else:
            prompt_emb = torch.cat(
                (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts
            prompt_emb = self.upsample_prompt(prompt_emb)

        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0, f"x size ({H}*{W}) are not even.")
        _assert(W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.reduction(x)
        x = self.norm(x)

        return x


class PromptedBasicLayer(nn.Module):
    def __init__(
            self, dim, input_resolution, depth, num_heads, window_size,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0,
            # add two more parameters for prompt
            num_prompts=None, prompt_location=None, deep_prompt=None, p_vk_cfg=None
            ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.grad_checkpointing = False
        
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        self.deep_prompt = deep_prompt
        self.p_vk_cfg = p_vk_cfg
        
        # build blocks
        
        if num_prompts is not None:
            self.blocks = nn.ModuleList([
                PromptedSwinTransformerBlock(
                    num_prompts, prompt_location,
                    dim=dim, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size)
                
                for i in range(depth)])
            self.deep_prompt = deep_prompt
            self.num_prompts = num_prompts
            self.prompt_location = prompt_location
            if self.deep_prompt and self.prompt_location != "prepend":
                raise ValueError("deep prompt mode for swin is only applicable to prepend")
        else:
            # print('should pass here for full fine-tuning')
            self.blocks =  nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size)
                for i in range(depth)])
            self.deep_prompt = False # should set false here for full fine-tuning

        # patch merging layer
        if downsample is not None:
            if num_prompts is None:
                self.downsample = downsample(
                    input_resolution, dim=dim, norm_layer=norm_layer
                )
            else:
                self.downsample = downsample(
                    num_prompts, prompt_location, deep_prompt,
                    input_resolution, dim=dim, norm_layer=norm_layer
                )
        else:
            self.downsample = nn.Identity()
            
    def forward(self, x, deep_prompt_embd=None):
        # print('111', self.deep_prompt)
        if self.deep_prompt and deep_prompt_embd is None:
            raise ValueError("need deep_prompt embddings")

        if not self.deep_prompt:
            for blk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
        else:
            # add the prompt embed before each blk call
            B = x.shape[0]  # batchsize
            num_blocks = len(self.blocks)
            if deep_prompt_embd.shape[0] != num_blocks:
                # first layer
                for i in range(num_blocks):
                    if i == 0:
                        x = self.blocks[i](x)

                    else:
                        prompt_emb = deep_prompt_embd[i-1].expand(B, -1, -1)
                        x = torch.cat(
                            (prompt_emb, x[:, self.num_prompts:, :]),
                            dim=1
                        )
                        x = self.blocks[i](x)
            else:
                # other layers
                for i in range(num_blocks):
                    prompt_emb = deep_prompt_embd[i].expand(B, -1, -1)
                    x = torch.cat(
                        (prompt_emb, x[:, self.num_prompts:, :]),
                        dim=1
                    )
                    x = self.blocks[i](x)

        x = self.downsample(x)
        return x
    
    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)
            
class PromptedSwinV2(SwinTransformerV2):
    def __init__(self, prompt_config, checkpoint_path=None,
                img_size=224, patch_size=(4,4), in_chans=3, num_classes=1000, global_pool='avg',
                embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                window_size=7, mlp_ratio=4., qkv_bias=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                pretrained_window_sizes=(0, 0, 0, 0), **kwargs):
        super(PromptedSwinV2, self).__init__(img_size, patch_size, in_chans, num_classes, global_pool,
                        embed_dim, depths, num_heads,
                        window_size, mlp_ratio, qkv_bias,
                        drop_rate, attn_drop_rate, drop_path_rate,
                        norm_layer, ape, patch_norm,
                        pretrained_window_sizes, **kwargs)
        if checkpoint_path:
            self._load_pretrained_weights(checkpoint_path)
        
        self.prompt_config = prompt_config
        self.num_prompts = self.prompt_config.NUM_PROMPTS
        
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
        self.ape = ape
        
        self._initialize_prompt_tokens(img_size, embed_dim, patch_size, depths)
        self._initialize_layers(img_size, patch_size, in_chans, num_classes, global_pool,
                                embed_dim, depths, num_heads,
                                window_size, mlp_ratio, qkv_bias,
                                drop_rate, attn_drop_rate, drop_path_rate,
                                norm_layer, ape, patch_norm,
                                pretrained_window_sizes, **kwargs)
        
        
    def _initialize_prompt_tokens(self, img_size, embed_dim, patch_size, depths):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))
        
        if self.prompt_config.LOCATION == "below":
            self.patch_embed.proj = Conv2d(
                in_channels=self.num_prompts + 3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)
            self.prompt_tokens = nn.Parameter(torch.zeros(1, self.num_prompts, img_size[0], img_size[1]))
            nn.init.uniform_(self.prompt_tokens.data, -val, val)
        
        elif self.prompt_config.LOCATION == "pad":
            self.prompt_tokens_tb = nn.Parameter(torch.zeros(1, 3, 2 * self.num_prompts, img_size[0]))
            self.prompt_tokens_lr = nn.Parameter(torch.zeros(1, 3, img_size[0] - 2 * self.num_prompts, 2 * self.num_prompts))
            nn.init.uniform_(self.prompt_tokens_tb.data, 0.0, 1.0)
            nn.init.uniform_(self.prompt_tokens_lr.data, 0.0, 1.0)
            self.prompt_norm = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        else:
            # for "prepend"
            self.prompt_tokens = nn.Parameter(torch.zeros(1, self.num_prompts, embed_dim))
            nn.init.uniform_(self.prompt_tokens.data, -val, val)
            
            if self.prompt_config.DEEP:
                self.deep_prompt_tokens = nn.ParameterList([
                    nn.Parameter(torch.zeros(depths[i], self.num_prompts, embed_dim * (2 ** i)))
                    for i in range(len(depths))
                ])
                for token in self.deep_prompt_tokens:
                    nn.init.uniform_(token.data, -val, val)
            else:
                self.deep_prompt_tokens = None

    def _initialize_layers(self, img_size, patch_size, in_chans, num_classes, global_pool,
                           embed_dim, depths, num_heads,
                           window_size, mlp_ratio, qkv_bias,
                           drop_rate, attn_drop_rate, drop_path_rate,
                           norm_layer, ape, patch_norm,
                           pretrained_window_sizes, **kwargs):
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PromptedBasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patch_embed.grid_size[0] // (2 ** i_layer),
                    self.patch_embed.grid_size[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PromptedPatchMerging if (i_layer < self.num_layers - 1) else None,
                pretrained_window_size=pretrained_window_sizes[i_layer],
                num_prompts=self.num_prompts,
                prompt_location=self.prompt_config.LOCATION,
                deep_prompt=self.prompt_config.DEEP
            )
            self.layers.append(layer)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _load_pretrained_weights(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and "prompt_tokens" not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
    
    def _get_patch_embeddings(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        return self.pos_drop(x)
    
    def incorporate_prompt(self, x):
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            x = self._get_patch_embeddings(x)
            prompt_embd = self.prompt_dropout(self.prompt_tokens.expand(B, -1, -1))
            x = torch.cat((prompt_embd, x), dim=1)
        elif self.prompt_config.LOCATION == "add":
            x = self._get_patch_embeddings(x)
            x = x + self.prompt_dropout(self.prompt_tokens.expand(B, -1, -1))
        elif self.prompt_config.LOCATION == "below":
            x = torch.cat((x, self.prompt_norm(self.prompt_tokens).expand(B, -1, -1, -1)), dim=1)
            x = self._get_patch_embeddings(x)
        else:
            raise ValueError("Other prompt locations are not supported")
        return x
    
    def forward_features(self, x):
        x = self.incorporate_prompt(x)
        if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
            for layer, deep_prompt_embd in zip(self.layers, self.deep_prompt_tokens):
                x = layer(x, self.prompt_dropout(deep_prompt_embd))
        else:
            for layer in self.layers:
                x = layer(x)
        return self.norm(x)
