# Adaptive Pyramid Context Network for Semantic Segmentation



| **ackbone**        | **Resnet101_v1c** |
| ------------------ | ----------------- |
| **Decode Head**    | **APCHead**       |
| **Auxiliary Head** | **FCNHead**       |

![source](./img/1.png)![gt](./img/1-gt.png)![pre](./img/1-pre.png)

***
![source](./img/2.png)![gt](./img/2-gt.png)![pre](./img/2-pre.png)
## Environment

```shell
#torch 1.8.0
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
pip install prettytable
#paddle
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```





## Pretrained model：

```yml
#mmcv resnet101_v1c  mmcv/model_zoo/open_mmlab.json:
"resnet101_v1c": "https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth"
```



## Weight tranpose：

###参数对齐日志：

```shell
cd paddle_apcnet/architectures
cp apcnetxxx_torch.pth paddle_apcnet/architectures/pretrained/apcnetxxx_torch.pth
python torchModel2pdModel.py #generate apcnetxxx_paddle.pdparams
```
[百度网盘](https://pan.baidu.com/s/189PaYKvoN2SSQLisOoTvqw)
提取码: mrla 


## Dataset：

### Cityscapes:

gtFine 5000

1. Downdown to ./dataset

2. sh prepare.sh
3. cd cityscapesscripts/preparation
4. python createTrainIdLabelImgs.py





## Train:

Paddle:

```shell
cd paddle_apcnet
sh paddle.sh
```



Torch:

```shell
cd torch_apcnet
sh torch.sh
```



## Test:

Paddle:

```shell
cd paddle_apcnet
python test.py
```





## Measure：

| Model               | mIou   |
| ------------------- | ------ |
| APCNet(torch)       | 79.08% |
| APCNet(paddle)(bilinear)      | 74.93% |
| APCNet(paddle)(BICUBIC)      | 73.14% |
| APCNet(paddle)(AREA)      | 72.22% |
| APCNet(paddle)(bilinear)(APCHead+FCNHead)| 73.21% |
| APCNet(paddle)(bilinear)(1.0APCHead+0.4FCNHead)| 73.49% |
| APCNet+CRFs(paddle) | 72.47% |



