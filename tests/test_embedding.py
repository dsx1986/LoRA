import pytest
import torch
import torch.nn as nn
from loralib.layers import Embedding

class TestEmbedding:
    """Tests for the Embedding layer with LoRA"""
    
    def test_init(self):
        """Test initialization of Embedding layer with LoRA"""
        # Test with LoRA (r > 0)
        lora_embedding = Embedding(num_embeddings=100, embedding_dim=20, r=4, lora_alpha=8)
        
        assert lora_embedding.r == 4
        assert lora_embedding.lora_alpha == 8
        assert lora_embedding.scaling == 2.0  # lora_alpha / r
        assert lora_embedding.weight.requires_grad is False  # Weight should be frozen
        assert hasattr(lora_embedding, 'lora_A')
        assert hasattr(lora_embedding, 'lora_B')
        assert lora_embedding.lora_A.shape == (4, 100)
        assert lora_embedding.lora_B.shape == (20, 4)
        
        # Test without LoRA (r = 0)
        regular_embedding = Embedding(num_embeddings=100, embedding_dim=20, r=0)
        
        assert regular_embedding.r == 0
        assert not hasattr(regular_embedding, 'lora_A')
        assert not hasattr(regular_embedding, 'lora_B')
        assert regular_embedding.weight.requires_grad is True  # Weight should be trainable
    
    def test_forward(self):
        """Test forward pass of Embedding layer with LoRA"""
        # Create an Embedding layer with LoRA
        lora_embedding = Embedding(num_embeddings=100, embedding_dim=20, r=4, lora_alpha=8)
        
        # Set weights to known values for testing
        nn.init.zeros_(lora_embedding.weight)
        nn.init.ones_(lora_embedding.lora_A)
        nn.init.ones_(lora_embedding.lora_B)
        
        # Create input tensor (indices)
        x = torch.tensor([1, 5, 10])
        
        # Expected output: regular output + LoRA adjustment
        # Regular output is zeros (weight is zeros)
        # LoRA adjustment: F.embedding(x, A.T) @ B.T * scaling
        # With ones in A and B, this becomes: ones(3, 4) @ ones(4, 20) * 2.0
        # = ones(3, 20) * 4 * 2.0
        # = ones(3, 20) * 8.0
        expected = torch.ones(3, 20) * 8.0
        
        # Get actual output
        output = lora_embedding(x)
        
        # Check if output matches expected
        assert torch.allclose(output, expected)
    
    def test_merge_weights(self):
        """Test merging of weights in Embedding layer with LoRA"""
        # Create an Embedding layer with LoRA
        lora_embedding = Embedding(num_embeddings=100, embedding_dim=20, r=4, lora_alpha=8, merge_weights=True)
        
        # Set weights to known values for testing
        nn.init.zeros_(lora_embedding.weight)
        nn.init.ones_(lora_embedding.lora_A)
        nn.init.ones_(lora_embedding.lora_B)
        
        # Switch to eval mode to trigger weight merging
        lora_embedding.eval()
        
        # Check if weights are merged
        assert lora_embedding.merged is True
        
        # Expected merged weight: original weight + (B @ A).T * scaling
        # With zeros in weight and ones in A and B, this becomes: 0 + (ones(20, 4) @ ones(4, 100)).T * 2.0
        # = ones(100, 20) * 4 * 2.0
        # = ones(100, 20) * 8.0
        expected_weight = torch.ones(100, 20) * 8.0
        
        # Check if weight matches expected
        assert torch.allclose(lora_embedding.weight, expected_weight)
        
        # Switch back to train mode
        lora_embedding.train()
        
        # Check if weights are unmerged
        assert lora_embedding.merged is False
        assert torch.allclose(lora_embedding.weight, torch.zeros(100, 20))
