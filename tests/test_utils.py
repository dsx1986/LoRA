import pytest
import torch
import torch.nn as nn
from loralib.utils import mark_only_lora_as_trainable, lora_state_dict
from loralib.layers import Linear, LoRALayer


class TestUtils:
    """Tests for the utility functions in loralib"""

    def test_mark_only_lora_as_trainable(self):
        """Test mark_only_lora_as_trainable function"""

        # Create a simple model with LoRA layers
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = Linear(10, 20, r=4)  # With LoRA
                self.linear2 = nn.Linear(20, 30)  # Regular linear
                self.linear3 = Linear(30, 40, r=0)  # Without LoRA (r=0)
                self.bias = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x + self.bias

        model = SimpleModel()

        # Make sure all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True

        # Verify all parameters are trainable
        for name, param in model.named_parameters():
            assert param.requires_grad is True

        # Mark only LoRA parameters as trainable with bias='none'
        mark_only_lora_as_trainable(model, bias="none")

        # Check which parameters are trainable
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad is True, f"{name} should be trainable"
            else:
                assert param.requires_grad is False, f"{name} should not be trainable"

        # Reset all parameters to trainable
        for param in model.parameters():
            param.requires_grad = True

        # Mark only LoRA parameters as trainable with bias='all'
        mark_only_lora_as_trainable(model, bias="all")

        # Check which parameters are trainable
        for name, param in model.named_parameters():
            if "lora_" in name or "bias" in name:
                assert param.requires_grad is True, f"{name} should be trainable"
            else:
                assert param.requires_grad is False, f"{name} should not be trainable"

    def test_lora_state_dict(self):
        """Test lora_state_dict function"""

        # Create a simple model with LoRA layers
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = Linear(10, 20, r=4)  # With LoRA
                self.linear2 = nn.Linear(20, 30)  # Regular linear
                self.linear3 = Linear(30, 40, r=0)  # Without LoRA (r=0)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        model = SimpleModel()

        # Get LoRA state dict with bias='none'
        lora_dict = lora_state_dict(model, bias="none")

        # Check that only LoRA parameters are included
        for name in lora_dict:
            assert "lora_" in name, f"{name} should contain 'lora_'"
            assert "bias" not in name, f"{name} should not contain 'bias'"

        # Get LoRA state dict with bias='all'
        lora_dict_with_bias = lora_state_dict(model, bias="all")

        # Check that LoRA parameters and all biases are included
        for name in lora_dict_with_bias:
            assert (
                "lora_" in name or "bias" in name
            ), f"{name} should contain 'lora_' or 'bias'"
