import paddle
import torchvision
import torch
import paddle.nn as nn
import numpy as np
import os
def trans():
    path='./resnet_pytorch.pth'
    model=torchvision.models.resnet101(num_classes=1000)
    torch_dict=torch.load(path)
    model.load_state_dict(torch_dict['net'])
    paddle_dict = {}
    # paddleModel=paddle.vision.models.resnet101()
    for key in torch_dict['net']:
        weight=torch_dict['net'][key].cpu().detach().numpy()  
        paddle_dict[key]=weight

    # paddle.save(paddle_dict,'./resnet_paddle.pdparams')

def seePaddle():
    model=paddle.vision.models.resnet101()
    for ele in model.state_dict():
        print(ele)
    
def seePytorch():
    path='./resnet_pytorch.pth'
    torch_dict=torch.load(path)
    model=torchvision.models.resnet101(num_classes=1000)
    model.load_state_dict(torch_dict['net'])
    for ele in model.state_dict():
        print(ele)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)
class Bottleneck(nn.Layer):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out
class ResNet(nn.Layer):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2D(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # print(block.expansion)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2D):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2D):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias_attr, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
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
        
        #[1, 2048, 24, 24]
        return x
def resnet101(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        path=os.path.join(os.path.split(os.path.realpath(__file__))[0],'./pretrained/resnet101.pdparams')
        param=paddle.load(path)
        model.set_state_dict(param)
        msg='----resnet101 pretrained load success from {}----'.format(path)
        # model.set_dict(param)
    else:
        msg='----resnet101 pretrained load FALSE!!!----'
    return model,msg
if __name__=='__main__':
    # models={}
    # models,msg1=resnet101()
    model,msg=resnet101()
    # x=paddle.normal(shape=[10,3,512,1024])
    # out=model(x)
    # print(out.shape)
    # print(model.keys())
    # model=resnet101()
    # data_1 = np.random.rand(1, 3, 768, 768).astype(np.float32)
    # data_1=paddle.to_tensor(data_1)
    # out=model(data_1)
    # print('----')
    # print(out.shape)
    # seePaddle()
    # print('----')
    # seePytorch()