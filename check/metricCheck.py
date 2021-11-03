import numpy as np
import paddle
import torch

from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
def getMiou(mat, labels_num=19):
    ious = []
    for i in range(labels_num):
        iou_i = mat[i][i] / (np.sum(mat[i], axis=0) + np.sum(mat[:, i], axis=0) - mat[i][i])
        ious.append(iou_i)

    res = (np.mean(ious), ious)
    return res
def getConfusionMatrix(prediction, target, ignore_label=255):
    if ignore_label:
        confusionMatrix = np.zeros((19, 19), dtype=int)
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
        for (p1, p2) in zip(target, prediction):
            if p1 != ignore_label:
                confusionMatrix[p1, p2] += 1
        return confusionMatrix
    else:
        confusionMatrix = np.zeros((20, 20), dtype=int)
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
        for (p1, p2) in zip(target, prediction):
            if p1 != 255:
                confusionMatrix[p1, p2] += 1
            elif p1 == 255:
                confusionMatrix[19, 19] += 1
        return confusionMatrix

    # print(confusionMatrix.shape)
if __name__=="__main__":
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    fake_pred=np.random.randint(0,19,(5,512,1024))
    fake_target=np.random.randint(0,19,(5,512,1024))

    pred=paddle.to_tensor(fake_pred)
    target = paddle.to_tensor(fake_target)

    pred=pred.numpy()
    target=target.numpy()
    ignore_label = 255
    if ignore_label:
        confusionMatrix = np.zeros((19, 19))
    else:
        confusionMatrix = np.zeros((20, 20))
    confusionMatrix+=getConfusionMatrix(pred, target, ignore_label=255)
    miou,_=getMiou(confusionMatrix)


    reprod_log_1.add("miou", np.array(miou))
    reprod_log_1.save("loss_paddle.npy")

    pred = torch.from_numpy(fake_pred).cuda()
    target = torch.from_numpy(fake_target).cuda()
    pred = pred.data.cpu().numpy()
    target = target.data.cpu().numpy()
    ignore_label = 255
    if ignore_label:
        confusionMatrix = np.zeros((19, 19))
    else:
        confusionMatrix = np.zeros((20, 20))
    confusionMatrix += getConfusionMatrix(pred, target, ignore_label=255)
    miou, _ = getMiou(confusionMatrix)


    reprod_log_2.add("miou",np.array(miou))
    reprod_log_2.save("loss_torch.npy")

    diff_helper = ReprodDiffHelper()
    info1 = diff_helper.load_info("./loss_paddle.npy")
    info2 = diff_helper.load_info("./loss_torch.npy")
    diff_helper.compare_info(info1, info2)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff-loss.txt")
