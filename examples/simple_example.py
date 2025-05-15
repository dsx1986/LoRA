"""
Simple example demonstrating how to use LoRA with a PyTorch model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import loralib as lora

# Create a simple model with LoRA layers
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5, r=4):
        super().__init__()
        # Replace nn.Linear with lora.Linear
        self.fc1 = lora.Linear(input_size, hidden_size, r=r)
        self.act = nn.ReLU()
        # Regular nn.Linear (not using LoRA)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Another LoRA layer
        self.fc3 = lora.Linear(hidden_size, output_size, r=r)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

def main():
    # Create a model
    model = SimpleModel()
    
    # Mark only LoRA parameters as trainable
    lora.mark_only_lora_as_trainable(model)
    
    # Print parameter status
    print("Parameter status:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    
    # Create dummy data
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)
    
    # Train the model
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("\nTraining for 10 epochs...")
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Save only the LoRA parameters
    lora_state_dict = lora.lora_state_dict(model)
    torch.save(lora_state_dict, "lora_checkpoint.pt")
    print("\nSaved LoRA parameters to lora_checkpoint.pt")
    
    # Switch to evaluation mode (merges LoRA weights)
    model.eval()
    print("\nSwitched to eval mode (LoRA weights are merged)")
    
    # Switch back to training mode (unmerges LoRA weights)
    model.train()
    print("Switched back to train mode (LoRA weights are unmerged)")

if __name__ == "__main__":
    main()
