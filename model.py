import torch.nn as nn


class steering_model(nn.Module):
    #reference: https://arxiv.org/abs/1604.07316
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5,5),stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5,5),stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5,5),stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3,3),stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152,100),
            nn.Linear(100,50),
            nn.Linear(50,10),
            nn.Linear(10,1)
        )
    def forward(self,inputs):
        x = self.backbone(inputs)
        return x
