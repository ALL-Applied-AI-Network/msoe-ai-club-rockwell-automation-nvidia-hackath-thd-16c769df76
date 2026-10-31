"""Compact recipe-conditioned vision model for vial carryover monitoring."""

from __future__ import annotations

import torch
from torch import nn


RECIPE_FEATURES = (
    "previous_t100_fraction",
    "previous_t200_fraction",
    "previous_t400_fraction",
    "current_t100_fraction",
    "current_t200_fraction",
    "current_t400_fraction",
    "expected_r",
    "expected_g",
    "expected_b",
    "queue_confidence",
)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class RecipeConditionedCarryoverNet(nn.Module):
    """Three-head model.

    risk_logits: low/medium/high transition risk, from the recipe branch.
    observed_logit: whether a visual carryover error is visible now.
    severity_logits: clean/mild/medium/severe visible condition.

    The final recipe feature is queue confidence. Setting it to zero and zeroing
    the other recipe features invokes the image-only fallback path.
    """

    def __init__(self, recipe_features: int = len(RECIPE_FEATURES), image_embedding: int = 128):
        super().__init__()
        self.image_encoder = nn.Sequential(
            ConvBlock(3, 24),
            ConvBlock(24, 48),
            ConvBlock(48, 96),
            ConvBlock(96, image_embedding),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.recipe_encoder = nn.Sequential(
            nn.Linear(recipe_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
        )
        self.risk_head = nn.Sequential(nn.Linear(64, 32), nn.SiLU(inplace=True), nn.Linear(32, 3))
        # Twelve explicit color features help the fusion heads compare the
        # rendered liquid with a recipe color they have never seen in training.
        fused = image_embedding + 64 + 12
        self.observed_head = nn.Sequential(
            nn.Linear(fused, 96), nn.SiLU(inplace=True), nn.Dropout(0.15), nn.Linear(96, 1)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(fused, 96), nn.SiLU(inplace=True), nn.Dropout(0.15), nn.Linear(96, 4)
        )

    def forward(self, image: torch.Tensor, recipe: torch.Tensor):
        image_embedding = self.image_encoder(image)
        recipe_embedding = self.recipe_encoder(recipe)
        image_01 = (image + 1.0) * 0.5
        height, width = image_01.shape[-2:]
        # The published FillCam ROI places the vial in the central stripe.
        region = image_01[:, :, : int(height * 0.92), int(width * 0.25) : int(width * 0.75)]
        region_mean = region.mean(dim=(2, 3))
        region_std = region.std(dim=(2, 3), unbiased=False)
        value = region.amax(dim=1, keepdim=True)
        chroma = value - region.amin(dim=1, keepdim=True)
        weights = (
            torch.sigmoid((chroma - 0.035) * 24.0)
            * torch.sigmoid((value - 0.06) * 24.0)
            * torch.sigmoid((0.96 - value) * 24.0)
        )
        weighted_color = (region * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp_min(1e-6)
        expected_color = recipe[:, 6:9]
        color_stats = torch.cat(
            (region_mean, region_std, weighted_color, weighted_color - expected_color), dim=1
        )
        fused = torch.cat((image_embedding, recipe_embedding, color_stats), dim=1)
        return {
            "risk_logits": self.risk_head(recipe_embedding),
            "observed_logit": self.observed_head(fused).squeeze(1),
            "severity_logits": self.severity_head(fused),
        }

