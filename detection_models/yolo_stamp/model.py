import torch
import torch.nn as nn

from .constants import *

"""
    Class for custom activation.
"""
class SymReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return torch.min(torch.max(input, -torch.ones_like(input)), torch.ones_like(input))

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


"""
    Class implementing YOLO-Stamp architecture described in https://link.springer.com/article/10.1134/S1054661822040046.
"""
class YOLOStamp(nn.Module):
    def __init__(
            self,
            anchors=ANCHORS,
            in_channels=3,
    ):
        super().__init__()
        
        self.register_buffer('anchors', torch.tensor(anchors))

        self.act = SymReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm3 = nn.BatchNorm2d(num_features=16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm4 = nn.BatchNorm2d(num_features=16)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm5 = nn.BatchNorm2d(num_features=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm6 = nn.BatchNorm2d(num_features=24)
        self.conv7 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm7 = nn.BatchNorm2d(num_features=24)
        self.conv8 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm8 = nn.BatchNorm2d(num_features=48)
        self.conv9 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm9 = nn.BatchNorm2d(num_features=48)
        self.conv10 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm10 = nn.BatchNorm2d(num_features=48)
        self.conv11 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm11 = nn.BatchNorm2d(num_features=64)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.norm12 = nn.BatchNorm2d(num_features=256)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=len(anchors) * 5, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
    
    def forward(self, x, head=True):
        x = x.type(self.conv1.weight.dtype)
        x = self.act(self.pool(self.norm1(self.conv1(x))))
        x = self.act(self.pool(self.norm2(self.conv2(x))))
        x = self.act(self.pool(self.norm3(self.conv3(x))))
        x = self.act(self.pool(self.norm4(self.conv4(x))))
        x = self.act(self.pool(self.norm5(self.conv5(x))))
        x = self.act(self.norm6(self.conv6(x)))
        x = self.act(self.norm7(self.conv7(x)))
        x = self.act(self.pool(self.norm8(self.conv8(x))))
        x = self.act(self.norm9(self.conv9(x)))
        x = self.act(self.norm10(self.conv10(x)))
        x = self.act(self.norm11(self.conv11(x)))
        x = self.act(self.norm12(self.conv12(x)))
        x = self.conv13(x)
        nb, _, nh, nw= x.shape
        x = x.permute(0, 2, 3, 1).view(nb, nh, nw, self.anchors.shape[0], 5)
        return x
