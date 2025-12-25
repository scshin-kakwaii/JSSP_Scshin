import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True 
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.layer_norms = torch.nn.ModuleList() 
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                h = self.layer_norms[layer](h)
                h = F.relu(h)
            return self.linears[self.num_layers - 1](h)

class MLPActor(MLP):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLPActor, self).__init__(num_layers, input_dim, hidden_dim, output_dim)

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                h = self.layer_norms[layer](h)
                # --- FIX: Chỉ apply tanh lên h, không gọi lại layer ---
                h = torch.tanh(h) 
            return self.linears[self.num_layers - 1](h)

class MLPCritic(MLP):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLPCritic, self).__init__(num_layers, input_dim, hidden_dim, output_dim)

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                h = self.layer_norms[layer](h)
                # --- FIX: Chỉ apply tanh lên h ---
                h = torch.tanh(h) 
            return self.linears[self.num_layers - 1](h)