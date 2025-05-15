import pytest
import torch
import torch.nn as nn
from loralib.layers import LoRALayer

class TestLoRALayer:
    """Tests for the base LoRALayer class"""
    
    def test_init(self):
        """Test initialization of LoRALayer"""
        lora_layer = LoRALayer(r=8, lora_alpha=16, lora_dropout=0.1, merge_weights=True)
        
        assert lora_layer.r == 8
        assert lora_layer.lora_alpha == 16
        assert lora_layer.merged is False
        assert lora_layer.merge_weights is True
        
    def test_dropout(self):
        """Test dropout functionality"""
        # Test with dropout
        lora_layer = LoRALayer(r=8, lora_alpha=16, lora_dropout=0.1, merge_weights=True)
        assert isinstance(lora_layer.lora_dropout, nn.Dropout)
        assert lora_layer.lora_dropout.p == 0.1
        
        # Test without dropout
        lora_layer = LoRALayer(r=8, lora_alpha=16, lora_dropout=0.0, merge_weights=True)
        # Should be a lambda function that returns the input
        x = torch.randn(5, 5)
        assert torch.equal(lora_layer.lora_dropout(x), x)
