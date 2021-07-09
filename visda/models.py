import torch.nn as nn
import torch
from torchvision import models


class DenseNet(nn.Module):

    def __init__(self, n_class):
        super(DenseNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.classif = nn.Linear(256, n_class)
        self.supervision = nn.Linear(256, 32)


    def forward(self, x):
        x = self.layers(x)
        classif = self.classif(x)
        supervision = self.supervision(x)
        return classif, supervision


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        resnet = models.resnet18(pretrained=True)
        resnet.eval()
        self.upsample = nn.Upsample((224, 224))
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def forward(self, x):

        with torch.no_grad():
            x = self.upsample(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class FullNet(nn.Module):

    def __init__(self, n_class):
        super(FullNet, self).__init__()

        self.resnet = ResNet()
        self.densenet = DenseNet(n_class)

    def forward(self, x):
        x = self.resnet(x)
        classif, supervision = self.densenet(x)
        return classif, supervision
