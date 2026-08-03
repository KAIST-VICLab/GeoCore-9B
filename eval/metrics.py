import torch
import torch.nn as nn
import numpy as np
import cv2
import lpips


class TrainingMetrics(nn.Module):
    """Batched PSNR / SSIM / LPIPS for validation during training.

    SSIM follows the 3-D Gaussian formulation used by the reference implementation:
    an [H, W, C] image is treated as a volume and filtered with a separable
    11x11x11 Gaussian Conv3d.
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)

        self.ssim_kernel = self._generate_3d_gaussian_kernel().to(self.device)

        self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device)
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def _generate_3d_gaussian_kernel(self) -> nn.Conv3d:
        kernel_1d = cv2.getGaussianKernel(11, 1.5)  # (11, 1)
        window_2d = np.outer(kernel_1d, kernel_1d.T)  # (11, 11)
        kernel_3d = np.stack([window_2d * k for k in kernel_1d], axis=0)  # (11, 11, 11)

        conv3d = nn.Conv3d(
            1, 1, (11, 11, 11),
            stride=1, padding=(5, 5, 5), bias=False, padding_mode="replicate"
        )
        conv3d.weight.requires_grad_(False)
        conv3d.weight[0, 0] = torch.tensor(kernel_3d, dtype=torch.float32)
        return conv3d

    @torch.no_grad()
    def calculate_psnr(self, preds: torch.Tensor, targets: torch.Tensor, data_range=1.0) -> float:
        """preds, targets: [B, C, H, W]"""
        mse = torch.mean((preds - targets) ** 2, dim=[1, 2, 3])
        mse = torch.clamp(mse, min=1e-10)
        psnr = 20.0 * torch.log10(data_range / torch.sqrt(mse))
        return psnr.mean().item()

    @torch.no_grad()
    def calculate_ssim(self, preds: torch.Tensor, targets: torch.Tensor, data_range=1.0) -> float:
        """preds, targets: [B, C, H, W]"""
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        # [B, C, H, W] -> [B, 1, H, W, C] so Conv3d filters across (H, W, C)
        t1 = preds.permute(0, 2, 3, 1).unsqueeze(1)
        t2 = targets.permute(0, 2, 3, 1).unsqueeze(1)

        mu1 = self.ssim_kernel(t1)
        mu2 = self.ssim_kernel(t2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.ssim_kernel(t1 ** 2) - mu1_sq
        sigma2_sq = self.ssim_kernel(t2 ** 2) - mu2_sq
        sigma12 = self.ssim_kernel(t1 * t2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean().item()

    @torch.no_grad()
    def calculate_lpips(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Inputs are [0, 1]; the lpips package expects [-1, 1]."""
        preds_scaled = preds * 2.0 - 1.0
        targets_scaled = targets * 2.0 - 1.0

        loss = self.lpips_fn(preds_scaled, targets_scaled)
        return loss.mean().item()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        """preds, targets: [B, C, H, W] in [0, 1]."""
        assert preds.shape == targets.shape, "Shape mismatch between predictions and targets."

        # generative outputs can land slightly outside [0, 1]
        preds = torch.clamp(preds, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        return {
            "PSNR": self.calculate_psnr(preds, targets),
            "SSIM": self.calculate_ssim(preds, targets),
            "LPIPS": self.calculate_lpips(preds, targets),
        }
