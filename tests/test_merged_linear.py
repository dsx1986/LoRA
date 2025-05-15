import pytest
import torch
import torch.nn as nn
from loralib.layers import MergedLinear

class TestMergedLinear:
    """Tests for the MergedLinear layer with LoRA"""
    
    def test_init(self):
        """Test initialization of MergedLinear layer with LoRA"""
        # Test with LoRA enabled for all parts
        lora_merged = MergedLinear(
            in_features=10, 
            out_features=6, 
            r=4, 
            lora_alpha=8,
            enable_lora=[True, True, True]  # Enable for all 3 parts (out_features=6 divided into 3 parts of size 2)
        )
        
        assert lora_merged.r == 4
        assert lora_merged.lora_alpha == 8
        assert lora_merged.scaling == 2.0  # lora_alpha / r
        assert lora_merged.weight.requires_grad is False  # Weight should be frozen
        assert hasattr(lora_merged, 'lora_A')
        assert hasattr(lora_merged, 'lora_B')
        assert lora_merged.lora_A.shape == (12, 10)  # r * sum(enable_lora), in_features
        assert lora_merged.lora_B.shape == (6, 4)  # out_features, r
        
        # Test with LoRA enabled for only some parts
        lora_merged_partial = MergedLinear(
            in_features=10, 
            out_features=6, 
            r=4, 
            lora_alpha=8,
            enable_lora=[True, False, True]  # Enable for 2 out of 3 parts
        )
        
        assert lora_merged_partial.lora_A.shape == (8, 10)  # r * sum(enable_lora), in_features
        assert lora_merged_partial.lora_B.shape == (4, 4)  # out_features // len(enable_lora) * sum(enable_lora), r
        
        # Test lora_ind tensor
        expected_ind = torch.tensor([True, True, False, False, True, True]).view(-1)
        assert torch.equal(lora_merged_partial.lora_ind, expected_ind)
    
    def test_zero_pad(self):
        """Test zero_pad function in MergedLinear"""
        lora_merged = MergedLinear(
            in_features=10, 
            out_features=6, 
            r=4, 
            lora_alpha=8,
            enable_lora=[True, False, True]  # Enable for 2 out of 3 parts
        )
        
        # Create a tensor to pad
        x = torch.ones(4, 5)  # 4 is the number of enabled positions (2 parts * 2 positions each)
        
        # Expected output: tensor with zeros at disabled positions
        expected = torch.zeros(6, 5)
        expected[0:2, :] = 1.0  # First part enabled
        expected[4:6, :] = 1.0  # Third part enabled
        
        # Get actual output
        output = lora_merged.zero_pad(x)
        
        # Check if output matches expected
        assert torch.equal(output, expected)
    
    def test_merge_AB(self):
        """Test merge_AB function in MergedLinear"""
        lora_merged = MergedLinear(
            in_features=10, 
            out_features=6, 
            r=4, 
            lora_alpha=8,
            enable_lora=[True, False, True]  # Enable for 2 out of 3 parts
        )
        
        # Set lora_A and lora_B to ones for testing
        nn.init.ones_(lora_merged.lora_A)
        nn.init.ones_(lora_merged.lora_B)
        
        # Get merged weights
        merged = lora_merged.merge_AB()
        
        # Expected shape
        assert merged.shape == (6, 10)
        
        # Check values - enabled parts should have non-zero values, disabled parts should be zero
        assert torch.all(merged[0:2, :] > 0)  # First part enabled
        assert torch.all(merged[2:4, :] == 0)  # Second part disabled
        assert torch.all(merged[4:6, :] > 0)  # Third part enabled
    
    def test_forward(self):
        """Test forward pass of MergedLinear layer with LoRA"""
        # Create a MergedLinear layer with LoRA
        lora_merged = MergedLinear(
            in_features=10, 
            out_features=6, 
            r=4, 
            lora_alpha=8,
            enable_lora=[True, False, True]  # Enable for 2 out of 3 parts
        )
        
        # Set weights to known values for testing
        nn.init.zeros_(lora_merged.weight)
        nn.init.zeros_(lora_merged.bias)
        nn.init.ones_(lora_merged.lora_A)
        nn.init.ones_(lora_merged.lora_B)
        
        # Create input tensor
        x = torch.ones(2, 10)
        
        # Get output
        output = lora_merged(x)
        
        # Check shape
        assert output.shape == (2, 6)
        
        # Check values - enabled parts should have non-zero values, disabled parts should be zero
        assert torch.all(output[:, 0:2] > 0)  # First part enabled
        assert torch.all(output[:, 2:4] == 0)  # Second part disabled
        assert torch.all(output[:, 4:6] > 0)  # Third part enabled
