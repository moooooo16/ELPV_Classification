import torch
import torchvision

class MyNeuralNetwork(torch.nn.module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=)
        )
    
    
    def forward(self, x):
        pass
    
    