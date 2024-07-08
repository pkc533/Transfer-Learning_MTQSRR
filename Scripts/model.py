import torch.nn as nn
import torchvision.models as torch_models

class feedNN(nn.Module):
    def __init__(self, input_dim=210):
        super(feedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 100)
        self.output = nn.Linear(100, 1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.output(out)
        
        return out