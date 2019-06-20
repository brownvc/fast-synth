import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from loc_dataset import *
from PIL import Image
import scipy.misc as m
import numpy as np
import math
import utils

"""
Module that predicts the location of the next object
"""

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, num_input_channels=17, use_fc=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x
        
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

class DownConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, stride=2, kernel_size=4, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.act = nn.LeakyReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(UpConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.act = nn.LeakyReLU()
    def forward(self, x):
        x = F.upsample(x, mode='nearest', scale_factor=2)
        return self.act(self.bn(self.conv(x)))

class Model(nn.Module):
    def __init__(self, num_classes, num_input_channels):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Dropout(p=0.2),
            resnet34(num_input_channels=num_input_channels),
            nn.Dropout(p=0.1),
            UpConvBlock(512, 256),
            UpConvBlock(256, 128),
            UpConvBlock(128, 64),
            UpConvBlock(64, 32),
            UpConvBlock(32, 16),
            UpConvBlock(16, 8),
            nn.Dropout(p=0.1),
            nn.Conv2d(8,num_classes,1,1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Location Training with Auxillary Tasks')
    parser.add_argument('--data-folder', type=str, default="bedroom_6x6", metavar='S')
    parser.add_argument('--num-workers', type=int, default=6, metavar='N')
    parser.add_argument('--last-epoch', type=int, default=-1, metavar='N')
    parser.add_argument('--train-size', type=int, default=6000, metavar='N')
    parser.add_argument('--save-dir', type=str, default="loc_test", metavar='S')
    parser.add_argument('--ablation', type=str, default=None, metavar='S')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N')
    parser.add_argument('--eps', type=float, default=1e-6, metavar='N')
    parser.add_argument('--centroid-weight', type=float, default=10, metavar="N")
    args = parser.parse_args()

    save_dir = args.save_dir
    utils.ensuredir(save_dir)
    batch_size = 16

    with open(f"data/{args.data_folder}/final_categories_frequency", "r") as f:
        lines = f.readlines()
    num_categories = len(lines)-2

    num_input_channels = num_categories+8

    logfile = open(f"{save_dir}/log_location.txt", 'w')
    def LOG(msg):
        print(msg)
        logfile.write(msg + '\n')
        logfile.flush()

    LOG('Building model...')
    model = Model(num_classes=num_categories+1, num_input_channels=num_input_channels)

    weight = [args.centroid_weight for i in range(num_categories+1)]
    weight[0] = 1
    print(weight)

    weight = torch.from_numpy(np.asarray(weight)).float().cuda()
    cross_entropy = nn.CrossEntropyLoss(weight=weight)
    softmax = nn.Softmax()

    LOG('Converting to CUDA...')
    model.cuda()
    cross_entropy.cuda()

    LOG('Building dataset...')
    train_dataset = LocDataset(
        data_root_dir = 'data',
        data_folder = args.data_folder,
        scene_indices = (0, args.train_size),
    )

    LOG('Building data loader...')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = args.num_workers,
        shuffle = True
    )

    LOG('Building optimizer...')
    optimizer = optim.Adam(model.parameters(),
        lr = args.lr,
        weight_decay = 2e-6,
    )

    if args.last_epoch < 0:
        load = False
        starting_epoch = 0
    else:
        load = True
        last_epoch = args.last_epoch

    if load:
        LOG('Loading saved models...')
        model.load_state_dict(torch.load(f"{save_dir}/location_{last_epoch}.pt"))
        optimizer.load_state_dict(torch.load(f"{save_dir}/location_optim_backup.pt"))
        starting_epoch = last_epoch + 1

    current_epoch = starting_epoch
    num_seen = 0

    model.train()
    LOG(f'=========================== Epoch {current_epoch} ===========================')

    def train():
        global num_seen, current_epoch
        for batch_idx, (data, target) \
                       in enumerate(train_loader):
            
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = cross_entropy(output,target)
            print(loss)
            
            loss.backward()
            optimizer.step()

            num_seen += batch_size
            if num_seen % 800 == 0:
                LOG(f'Examples {num_seen}/10000')
            if num_seen % 10000 == 0:
                LOG('Validating')
                num_seen = 0
                current_epoch += 1
                LOG(f'=========================== Epoch {current_epoch} ===========================')
                if current_epoch % 10 == 0:
                    torch.save(model.state_dict(), f"{save_dir}/location_{current_epoch}.pt")
                    torch.save(optimizer.state_dict(), f"{save_dir}/location_optim_backup.pt")

    while True:
        train()
