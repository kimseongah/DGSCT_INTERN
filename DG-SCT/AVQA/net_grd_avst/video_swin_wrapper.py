import torch
import torch.nn as nn
from torchvision.models.video import swin3d_b, Swin3D_B_Weights


class Swin3DWrapper(nn.Module):
    def __init__(self, pretrained=True, prompt_dim=128, num_prompts=10):
        super().__init__()
        weights = Swin3D_B_Weights.DEFAULT if pretrained else None
        self.backbone = swin3d_b(weights=weights)

        self.patch_embed = self.backbone.patch_embed
        self.norm = self.backbone.norm
        self.head = self.backbone.head

        # features는 stage / downsample 순
        self.layers = nn.ModuleList(self.backbone.features)

        self.prompt_per_stage = nn.ParameterList()

        resolutions = [(48, 48), (24, 24), (12, 12), (6, 6)]
        prompt_dims = [128, 256, 512, 1024]  # swin3d_b embedding dim
        i = 0
        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                (H_s, W_s), prompt_dim = resolutions[i], prompt_dims[i]
                i += 1
                prompt = nn.Parameter(torch.randn(1, num_prompts, H_s, W_s, prompt_dim))
                self.prompt_per_stage.append(prompt)
            else:
                self.prompt_per_stage.append(None)


    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, T_, H_, W_, C)
        B, T_, H_, W_, C = x.shape

        for idx, layer in enumerate(self.layers):
            if hasattr(self, 'prompt_per_stage') and self.prompt_per_stage[idx] is not None:
                prompt = self.prompt_per_stage[idx]  # (N_prompt, H_, W_, C)
                prompt = prompt.expand(B, -1, -1, -1, -1)  # → (B, N_prompt, H_, W_, C)
                x = torch.cat([prompt, x], dim=1)         # → (B, T_ + N_prompt, H_, W_, C)

            # Forward through the stage
            x = layer(x)

            if hasattr(self, 'prompt_per_stage') and self.prompt_per_stage[idx] is not None:
                x = x[:, -T_:, :, :, :]  # (B, T_, H_, W_, C)

        x = self.norm(x)  # (B, T_, H_, W_, C)
        return x


    def forward(self, x):
        x = self.forward_features(x)
        
        x = x.mean(dim=1)
        return self.head(x)
