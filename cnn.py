import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes=36, captcha_length=5):
        super(CNNModel, self).__init__()
        self.captcha_legth = captcha_length
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 32, )
    
