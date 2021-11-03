import torch,os
import paddle
import numpy as np
from resnet_paddle import ResNetV1C
from apcnet_paddle import APCHead
from fcnhead_paddle import FCNHead


def isIn(key, str):
    for ele in key:
        if ele in str:
            return True


def rename_state_dict(paddle_dict, torch_dict):
    result = {}
    _ = {}
    skip_params = ['batches_tracked', 'fc.weight', 'fc.bias']
    for k, v in torch_dict.items():
        if isIn(skip_params, k):
            print('---skip batches_tracked---')
        else:

            _[k] = v.cpu().detach().numpy()
    torch_dict = _

    assert len(torch_dict) == len(paddle_dict)
    for paddle_param, torch_param in zip(paddle_dict.items(), torch_dict.items()):
        k1, v1 = paddle_param
        k2, v2 = torch_param
        v1 = v1.numpy()
        v2 = v2
        print('{} shape {}, {} shape {}'.format(k1, v1.shape, k2, v2.shape))
        assert v1.shape == v2.shape
        result[k1] = v2
    print(len(torch_dict))
    print(len(paddle_dict))
    return result


def trans():
    path = './pretrained/resnet101_v1c-e67eebb6.pth'
    torch_dict = torch.load(path)
    paddle_dict = {}
    paddle_dict['state_dict'] = {}
    paddle_dict['meta'] = torch_dict['meta']

    model = ResNetV1C()
    paddle_dict['state_dict'] = rename_state_dict(model.state_dict(), torch_dict['state_dict'])


    paddle.save(paddle_dict, './pretrained/resnet101_aistudio.pdparams')
trans()