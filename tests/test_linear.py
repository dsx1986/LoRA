import pytest
import torch
import torch.nn as nn
import math
from loralib.layers import Linear

class TestLinear:
    """Tests for the Linear layer with LoRA"""
    
    def test_init(self):
        """Test initialization of Linear layer with LoRA"""
        # Test with LoRA (r > 0)
        lora_linear = Linear(in_features=10, out_features=5, r=4, lora_alpha=8)
        
        assert lora_linear.r == 4
        assert lora_linear.lora_alpha == 8
        assert lora_linear.scaling == 2.0  # lora_alpha / r
        assert lora_linear.weight.requires_grad is False  # Weight should be frozen
        assert hasattr(lora_linear, 'lora_A')
        assert hasattr(lora_linear, 'lora_B')
        assert lora_linear.lora_A.shape == (4, 10)
        assert lora_linear.lora_B.shape == (5, 4)
        
        # Test without LoRA (r = 0)
        regular_linear = Linear(in_features=10, out_features=5, r=0)
        
        assert regular_linear.r == 0
        assert not hasattr(regular_linear, 'lora_A')
        assert not hasattr(regular_linear, 'lora_B')
        assert regular_linear.weight.requires_grad is True  # Weight should be trainable
    
    def test_forward(self):
        """Test forward pass of Linear layer with LoRA"""
        # Create a Linear layer with LoRA
        lora_linear = Linear(in_features=10, out_features=5, r=4, lora_alpha=8)
        
        # Set weights to known values for testing
        nn.init.zeros_(lora_linear.weight)
        nn.init.zeros_(lora_linear.bias)
        nn.init.ones_(lora_linear.lora_A)
        nn.init.ones_(lora_linear.lora_B)
        
        # Create input tensor
        x = torch.ones(2, 10)
        
        # Expected output: regular output + LoRA adjustment
        # Regular output is zeros (weight is zeros)
        # LoRA adjustment: x @ A.T @ B.T * scaling
        # With ones in A and B, this becomes: x @ ones(4, 10).T @ ones(5, 4).T * 2.0
        # = x @ ones(10, 4) @ ones(4, 5) * 2.0
        # = ones(2, 10) @ ones(10, 4) @ ones(4, 5) * 2.0
        # = ones(2, 4) @ ones(4, 5) * 2.0
        # = ones(2, 5) * 10 * 4 * 2.0
        # = ones(2, 5) * 80.0
        expected = torch.ones(2, 5) * 80.0
        
        # Get actual output
        output = lora_linear(x)
        
        # Check if output matches expected
        assert torch.allclose(output, expected)
    
    def test_merge_weights(self):
        """Test merging of weights in Linear layer with LoRA"""
        # Create a Linear layer with LoRA
        lora_linear = Linear(in_features=10, out_features=5, r=4, lora_alpha=8, merge_weights=True)
        
        # Set weights to known values for testing
        nn.init.zeros_(lora_linear.weight)
        nn.init.ones_(lora_linear.lora_A)
        nn.init.ones_(lora_linear.lora_B)
        
        # Switch to eval mode to trigger weight merging
        lora_linear.eval()
        
        # Check if weights are merged
        assert lora_linear.merged is True
        
        # Expected merged weight: original weight + (B @ A) * scaling
        # With zeros in weight and ones in A and B, this becomes: 0 + (ones(5, 4) @ ones(4, 10)) * 2.0
        # = ones(5, 10) * 4 * 2.0
        # = ones(5, 10) * 8.0
        expected_weight = torch.ones(5, 10) * 8.0
        
        # Check if weight matches expected
        assert torch.allclose(lora_linear.weight, expected_weight)
        
        # Switch back to train mode
        lora_linear.train()
        
        # Check if weights are unmerged
        assert lora_linear.merged is False
        assert torch.allclose(lora_linear.weight, torch.zeros(5, 10))
