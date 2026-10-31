#!/usr/bin/env python3
"""Fast shape and backward-pass check; does not need dataset files."""

import torch

from model import RECIPE_FEATURES, RecipeConditionedCarryoverNet


def main():
    model = RecipeConditionedCarryoverNet()
    image = torch.randn(4, 3, 160, 160)
    recipe = torch.rand(4, len(RECIPE_FEATURES))
    output = model(image, recipe)
    assert output["risk_logits"].shape == (4, 3)
    assert output["observed_logit"].shape == (4,)
    assert output["severity_logits"].shape == (4, 4)
    loss = (output["risk_logits"].mean() + output["observed_logit"].mean()
            + output["severity_logits"].mean())
    loss.backward()
    print({"status": "passed", "parameters": sum(p.numel() for p in model.parameters())})


if __name__ == "__main__":
    main()

