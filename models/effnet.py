import torch
import torch.nn as nn
def make_divisible_by_8(v):
    new_v = max(8, int(v + 4) // 8 * 8)
    if new_v < 0.9 * v:
        new_v += 8
    return new_v

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, make_divisible_by_8(inp // reduction)),
            nn.SiLU(),
            nn.Linear(make_divisible_by_8(inp // reduction), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
def conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super().__init__()
        
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)
        if use_se:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                SELayer(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, config, limit):
        super().__init__()
        self.config = config
        input_channel = make_divisible_by_8(24)
        layers = [conv_3x3(3, input_channel, 2)]
        for t, c, n, s, use_se in config:
            output_channel = make_divisible_by_8(c)
            for i in range(n):
                layers.append(MBConv(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
            self.features = nn.Sequential(*layers)
            output_channel = 1792
            self.conv = conv_1x1(input_channel, output_channel)
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(output_channel, limit)
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def Effnetv2_s(limit=1000):
    config = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 1, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 1, 1],
        [6, 160,  9, 2, 1],
        [6, 256, 15, 1, 1],
    ]
    return EfficientNet(config, limit)