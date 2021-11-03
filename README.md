# Adaptive Pyramid Context Network for Semantic Segmentation


##Easy to Run

[AI Studio Link](https://aistudio.baidu.com/aistudio/projectdetail/2541370?contributionType=1&shared=1) @PaddlePaddle


| **ackbone**        | **Resnet101_v1c** |
| ------------------ | ----------------- |
| **Decode Head**    | **APCHead**       |
| **Auxiliary Head** | **FCNHead**       |


***
![source](./img/img.png)
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

###模型，日志,项目下载：

```shell
cd paddle_apcnet/architectures
cp apcnetxxx_torch.pth paddle_apcnet/architectures/pretrained/apcnetxxx_torch.pth
python torchModel2pdModel.py #generate apcnetxxx_paddle.pdparams
```
[Ai studio](https://aistudio.baidu.com/bdv3/user/303267/2541370/doc/tree/work/paddle_apcnet/experiments/apcnet-cityscapes)

[百度网盘](https://pan.baidu.com/s/1LVe__pSaNhcxxCQrLnQi8w)
提取码: ttqw 


模型对齐：参考./check/modelCheck 以及paddle_apcnet/architectures下的 参数转化文件，以及转换结果文件

loss对齐：参考./check/ 下的 lossCheck.py

miou对齐：参考./check/ 下的 metricCheck.py

训练对齐： 参考下载链接中的日志文件

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
直接跑apcnet训练好的转换后的torch模型，在val dataset 只有74.93%的miou


由于训练轮次只有一半， APCNet(paddle) 的miou为 79.28% %

| Model               | mIou   |
| ------------------- | ------ |
| APCNet(torch)       | 79.08% |
| APCNet(paddle)     | 79.28% |


