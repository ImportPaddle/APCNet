# Adaptive Pyramid Context Network for Semantic Segmentation

Cityscapes 79.08%

| Model          | Miou   |
| -------------- | ------ |
| APCNet(torch)  | 79.08% |
| APCNet(paddle) |        |



Backbone: Resnet101_v1c

Decode Head: APCHead 

Auxiliary Head: FCNHead

## 环境



```shell
#torch 1.8.0
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
pip install prettytable
#paddle
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```











```yml
#mmcv resnet101_v1c  mmcv/model_zoo/open_mmlab.json:
"resnet101_v1c": "https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth"
```



