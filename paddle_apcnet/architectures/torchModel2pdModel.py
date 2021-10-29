import torch,os
import paddle
import numpy as np
from resnet_paddle import ResNetV1C
from apcnet_paddle import APCHead
from fcnhead_paddle import FCNHead
def write_dict(state_dict,name):
    lines=[]
    for k,v in state_dict.items():
        if 'batches_tracked' in k:
            print('---skip--batches_tracked-')
            continue
        line=str(k)+'\t'+str(v.cpu().detach().numpy().shape)+'\n'
        # line=str(v)+'\t'+str(v.cpu().detach().numpy().shape)+'\n'
        lines.append(line)
    with open(name,'w')as f:
        f.writelines(lines)
def isIn(key,str):
    for ele in key:
        if ele in str:
            return True
def rename_state_dict(paddle_dict,torch_dict):
    result={}
    _={}
    skip_params=['batches_tracked','fc.weight','fc.bias']
    for k,v in torch_dict.items():
        if isIn(skip_params,k):
            print('---skip batches_tracked---')
        else:

            _[k]=v.cpu().detach().numpy()
    torch_dict=_
   
    assert len(torch_dict)==len(paddle_dict)
    for paddle_param,torch_param in zip(paddle_dict.items(),torch_dict.items()):
        k1,v1=paddle_param
        k2,v2=torch_param
        v1=v1.numpy()
        v2=v2
        print('{} shape {}, {} shape {}'.format(k1,v1.shape,k2,v2.shape))
        assert v1.shape==v2.shape
        result[k1]=v2
    print(len(torch_dict))
    print(len(paddle_dict))
    return result
def trans():
    path = './pretrained/resnet101_v1c-e67eebb6.pth'
    torch_dict = torch.load(path)
    print(torch_dict.keys())
    paddle_dict = {}
    paddle_dict['state_dict']={}
    paddle_dict['meta']=torch_dict['meta']
    # paddle_dict['state_dict']=torch_dict['state_dict']
    for key in torch_dict['state_dict']:
        weight = torch_dict['state_dict'][key].cpu().detach().numpy()
        # print(key)
        if key == 'fc.weight':
            weight = weight.transpose()
        key = key.replace('running_mean', '_mean')
        key = key.replace('running_var', '_variance')
        paddle_dict['state_dict'][key] = weight
    paddle.save(paddle_dict, './pretrained/resnet101_v1c-e67eebb6.pdparams')
    
        
    model=ResNetV1C()
    write_dict(model.state_dict(),'./paddleParams.txt')
    write_dict(torch_dict['state_dict'],'./torchParams.txt')
    
    
    paddle_dict['state_dict']=rename_state_dict(model.state_dict(),torch_dict['state_dict'])
    write_dict(torch_dict['state_dict'],'./rename_torchParams.txt')
    model.set_state_dict(paddle_dict['state_dict'])
    paddle.summary(model, (4, 3,512,1024))
    
    paddle.save(paddle_dict, './pretrained/resnet101_v1c-e67eebb6.pdparams')
    # paddle_dict=paddle.load('./pretrained/resnet101_v1c-e67eebb6.pdparams')
def see(path):
    state=torch.load(path)
    print(state.keys())
    # print(state['meta'])
    for k,v in state['meta'].items():
        print(k,v)
    
    backbone={}
    decode_head={}
    auxiliary_head={}
    all=len(state['state_dict'])
    print('all:{}'.format(all))
    for k,v in state['state_dict'].items():
        if 'backbone' in k:
            backbone[k]=v
        elif 'decode_head' in k:
            decode_head[k]=v
        elif 'auxiliary_head' in k:
            auxiliary_head[k]=v
    print('backbone:{}, decode_head:{},auxiliary_head:{}'.format(len(backbone),len(decode_head),len(auxiliary_head)))

def transEachDict(path):
    res={}
    
    state=torch.load(path)
    backbone_torch={}
    decode_head_torch={}
    auxiliary_head_torch={}
    all=len(state['state_dict'])
    print('all:{}'.format(all))
    for k,v in state['state_dict'].items():
        if 'backbone' in k:
            backbone_torch[k]=v
        elif 'decode_head' in k:
            decode_head_torch[k]=v
        elif 'auxiliary_head' in k:
            auxiliary_head_torch[k]=v
    
    backbone=ResNetV1C()
    write_dict(backbone.state_dict(),'./backbone_paddleParams.txt')
    write_dict(backbone_torch,'./backbone_torchParams.txt')
    paddle_dict=rename_state_dict(backbone.state_dict(),backbone_torch)
    backbone.set_state_dict(paddle_dict)
    print('load backbone success')
    
    
    apchead=APCHead()
    write_dict(apchead.state_dict(),'./apchead_paddleParams.txt')
    write_dict(decode_head_torch,'./apchead_torchParams.txt')
    paddle_dict=rename_state_dict(apchead.state_dict(),decode_head_torch)
    # write_dict(decode_head_torch,'./apchead_rename.txt')
    apchead.set_state_dict(paddle_dict)
    print('load apchead success')
    
    fcnhead=FCNHead()
    write_dict(fcnhead.state_dict(),'./fcnhead_paddleParams.txt')
    write_dict(auxiliary_head_torch,'./fcnhead_torchParams.txt')
    paddle_dict=rename_state_dict(fcnhead.state_dict(),auxiliary_head_torch)
    # write_dict(decode_head_torch,'./apchead_rename.txt')
    fcnhead.set_state_dict(paddle_dict)
    print('load fcnhead success')
    
    res['models']={}
    res['models']['backbone']=backbone.state_dict()
    res['models']['APCHead']=apchead.state_dict()
    res['models']['FCNHead']=fcnhead.state_dict()
    paddle.save(res,'./pretrained/apcnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.paparams')
if __name__=='__main__':
    # model=ResNetV1C()
    
    # paddle.summary(model, (4, 3,512,1024))
    # d1=dict(x1=1,x2=2)
    # d2=dict(y1=3,y2=4,y3=5,y4=5)
    # for p,t in zip(d1.items(),d2.items()):
    #     p_k,p_v=p
    #     t_k,t_v=t
    #     print('({},{}):({},{}))'.format(p_k,t_k,p_v,t_v))
    path='./pretrained/apcnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.pth'
    
    transEachDict(path)
    # path='./pretrained/apcnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.pth'
    # see(path)