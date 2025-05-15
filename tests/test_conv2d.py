import pytest
import torch
import torch.nn as nn
from loralib.layers import Conv2d


class TestConv2d:
    """Tests for the Conv2d layer with LoRA"""

    def test_init(self):
        """Test initialization of Conv2d layer with LoRA"""
        # Test with LoRA (r > 0)
        lora_conv = Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, r=4, lora_alpha=8
        )

        assert lora_conv.r == 4
        assert lora_conv.lora_alpha == 8
        assert lora_conv.scaling == 2.0  # lora_alpha / r
        assert lora_conv.conv.weight.requires_grad is False  # Weight should be frozen
        assert hasattr(lora_conv, "lora_A")
        assert hasattr(lora_conv, "lora_B")
        assert lora_conv.lora_A.shape == (
            12,
            9,
        )  # r * kernel_size, in_channels * kernel_size
        assert lora_conv.lora_B.shape == (48, 12)  # out_channels, r * kernel_size

        # Test without LoRA (r = 0)
        regular_conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, r=0)

        assert regular_conv.r == 0
        assert not hasattr(regular_conv, "lora_A")
        assert not hasattr(regular_conv, "lora_B")
        assert (
            regular_conv.conv.weight.requires_grad is True
        )  # Weight should be trainable

    def test_forward(self):
        """Test forward pass of Conv2d layer with LoRA"""
        # Create a Conv2d layer with LoRA
        lora_conv = Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, r=4, lora_alpha=8
        )

        # Create input tensor
        x = torch.randn(2, 3, 32, 32)

        # Get output shape
        output = lora_conv(x)

        # Check output shape (should be same as regular Conv2d)
        expected_shape = nn.Conv2d(3, 16, 3)(x).shape
        assert output.shape == expected_shape

    def test_merge_weights(self):
        """Test merging of weights in Conv2d layer with LoRA"""
        # Create a Conv2d layer with LoRA
        lora_conv = Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            r=4,
            lora_alpha=8,
            merge_weights=True,
        )

        # Get original weight
        original_weight = lora_conv.conv.weight.clone()

        # Switch to eval mode to trigger weight merging
        lora_conv.eval()

        # Check if weights are merged
        assert lora_conv.merged is True

        # Switch back to train mode
        lora_conv.train()

        # Check if weights are unmerged
        assert lora_conv.merged is False
        assert torch.allclose(lora_conv.conv.weight, original_weight)
