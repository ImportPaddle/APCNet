

import paddle
import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
np.random.seed(1)
import torch

def getLoss(pre,target,flag='torch'):
    if flag=='torch':
        pre=torch.from_numpy(pre).cuda()
        target=torch.from_numpy(target).cuda()
        criterion=torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        res=criterion(pre,target)
        return res.data.cpu().numpy()
    else:
        pre=paddle.to_tensor(pre)
        target=paddle.to_tensor(target)
        criterion=paddle.nn.CrossEntropyLoss(ignore_index=255, reduction='mean', axis=1)
        res=criterion(pre,target)
        return res.numpy()

def main():
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    # shapes=[1,10,32,32]
    pre_1=np.random.randn(5,(5,19,512,1024))
    target_1=np.random.randint(0,19,(5,512,1024))


    pytorch_res_1=getLoss(pre_1,target_1)
    paddle_res_1=getLoss(pre_1,target_1,flag='paddle')

    reprod_log_1.add("loss1", pytorch_res_1)
    reprod_log_1.save("loss_torch.npy")

    reprod_log_2.add("loss1", paddle_res_1)
    reprod_log_2.save("loss_paddle.npy")

def check():
    diff_helper = ReprodDiffHelper()
    info1 = diff_helper.load_info("./loss_paddle.npy")
    info2 = diff_helper.load_info("./loss_torch.npy")

    diff_helper.compare_info(info1,info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff-loss.txt")
if __name__=="__main__":    
    main()
    check()