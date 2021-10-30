from tqdm import tqdm
import paddle
import datetime
import logging
import os
from dataloader import CityScapesDataset
from architectures import getApcNet
from paddle.io import Dataset, DataLoader
from config import CONFIG
import paddle.nn.functional as F
import warnings
import glob
import numpy as np

warnings.filterwarnings('ignore')


class Trainer():
    def __init__(self, expName, resume=True, resume_inter='latest', config=None):
        super(Trainer, self).__init__()
        self.max_inter = int(80000 * 8 / CONFIG['train_batch_size'])
        self.inter = 1
        self.optimizers_step_size = int(self.max_inter / 21)
        self.config = CONFIG
        self.root = os.getcwd()
        self.resume = resume
        self.resume_inter = resume_inter

        self.exp_dir = self.init_exp_dir(expName=expName)
        self.logger = self.init_logger()
        self.models = self.init_models()
        self.dataloaders = self.init_dataloaders()
        self.optimizers = self.init_optimizers()
        self.criterions = self.init_criterions()
        self.init_check()
        self.stepEachEpoch = (len(self.dataloaders['train']) + 1)
        self.train_display_step = 50
        self.train_save_step = 2000
        self.val_step = 1

        self.logger.info('init complete')

    def run(self):
        ##resume
        if self.resume:
            self.resume_experiment()
        self.Loss = {}
        self.Loss['step_cnt'] = 0
        self.Loss['apc_loss'] = 0.0
        self.Loss['fcn_loss'] = 0.0

        while (self.inter < self.max_inter):
            for batch_id, batch in enumerate(tqdm(self.dataloaders['train']())):

                lossapc, lossfcn = self.train_step(batch_id, batch)
                self.Loss['step_cnt'] += 1
                self.Loss['apc_loss'] += lossapc
                self.Loss['fcn_loss'] += lossfcn

                if self.inter % self.train_display_step == 0:
                    self.logger.info('inter{} , APC Loss:{} FCN Loss:{},lr:{}' \
                                     .format(self.inter,
                                             self.Loss['apc_loss'] / self.Loss['step_cnt'],
                                             self.Loss['fcn_loss'] / self.Loss['step_cnt'],
                                             self.optimizers['APCHead'].get_lr(),
                                             )
                                     )
                    self.Loss['step_cnt'] = 0
                    self.Loss['apc_loss'] = 0.0
                    self.Loss['fcn_loss'] = 0.0
                if self.inter % self.train_save_step == 0:
                    self.save_checkpoint()
                    self.delete_checkpoint()

                if self.inter % self.val_step == 0:
                    self.val()
                self.inter += 1

    def resume_experiment(self):
        self.load_checkpoint()
        for i in range(self.inter):
            self.scheduler['backbone'].step()
            self.scheduler['APCHead'].step()
            self.scheduler['FCNHead'].step()
    def init_exp_dir(self, expName):
        expdir = os.path.join(self.root, 'experiments', expName)
        if os.path.exists(expdir) and not self.resume:
            # now_str = datetime.datetime.now().__str__().replace(' ','_')
            # expdir=os.path.join(self.root,'experiments',expName+'_'+now_str)
            # if os.path.exists(expdir):
            print('exist exp dir {}'.format(expdir))
            exit(1)
            # else:
            #     os.makedirs(expdir)
        else:

            os.makedirs(expdir, exist_ok=True)
            models_dir = os.path.join(expdir, 'ckpt')
            os.makedirs(models_dir, exist_ok=True)
        return expdir

    def init_logger(self):
        logger = logging.getLogger(__name__)
        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        logger.addHandler(strHandler)
        logger.setLevel(logging.INFO)
        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)
        now_str = datetime.datetime.now().__str__().replace(' ', '_')
        self.log_file = os.path.join(log_dir, 'LOG_INFO_' + now_str + '.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        logger.addHandler(self.log_fileHandler)
        return logger

    def init_models(self):
        networks, msg_resnet = getApcNet()
        pre_apcent = 0
        if pre_apcent:
            state = paddle.load(
                './architectures/pretrained/apcnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.pdparams')
            networks['backbone'].set_state_dict(state['models']['backbone'])
            networks['APCHead'].set_state_dict(state['models']['APCHead'])
            networks['FCNHead'].set_state_dict(state['models']['FCNHead'])
        self.logger.info(msg_resnet)
        return networks  ##['backbone','APCHead','FCNHead']
        pass

    def init_dataloaders(self):
        # trainLoader,valLoader,testLoader
        trainset = CityScapesDataset(root='../dataset/cityscapes', mode='train', SEED=1)
        valset = CityScapesDataset(root='../dataset/cityscapes', mode='val', SEED=1)
        testset = CityScapesDataset(root='../dataset/cityscapes', mode='test', SEED=1)
        trainLoader = DataLoader(trainset, batch_size=self.config['train_batch_size'], drop_last=False,
                                 num_workers=self.config['num_workers'], shuffle=True, use_buffer_reader=True)
        valLoader = DataLoader(valset, batch_size=self.config['val_batch_size'], drop_last=False,
                               num_workers=self.config['num_workers'], shuffle=True, use_buffer_reader=True)
        testLoader = DataLoader(testset, batch_size=self.config['test_batch_size'], drop_last=False,
                                num_workers=self.config['num_workers'], shuffle=True, use_buffer_reader=True)
        print(len(trainset))
        print(len(valset))
        print(len(testset))

        assert len(trainset) + len(valset) + len(testset) == 5000, 'images number is not 5000'
        assert len(trainset) == 2975
        assert len(valset) == 500
        assert len(testset) == 1525
        dataloaders = {}
        dataloaders['train'] = trainLoader
        dataloaders['val'] = valLoader
        dataloaders['test'] = testLoader
        return dataloaders

    def init_optimizers(self):

        optimizers = {}
        self.scheduler = {}
        backboneCfg = self.config['optimizers']['backbone']

        self.scheduler['backbone'] = paddle.optimizer.lr.StepDecay(learning_rate=backboneCfg['lr'],
                                                                   step_size=self.optimizers_step_size, gamma=0.9,
                                                                   verbose=False)
        optimizers['backbone'] = paddle.optimizer.Momentum(
            parameters=self.models['backbone'].parameters(),
            learning_rate=self.scheduler['backbone'],
            momentum=backboneCfg['momentum'],
            weight_decay=backboneCfg['weight_decay'])
        # paddle.optimizer.SGD(learning_rate=0.001, parameters=None, weight_decay=None)
        apcnetCfg = self.config['optimizers']['APCHead']

        # step_size=5
        self.scheduler['APCHead'] = paddle.optimizer.lr.StepDecay(learning_rate=apcnetCfg['lr'],
                                                                  step_size=self.optimizers_step_size, gamma=0.9,
                                                                  verbose=False)
        optimizers['APCHead'] = paddle.optimizer.Momentum(
            parameters=self.models['APCHead'].parameters(),
            learning_rate=self.scheduler['APCHead'],
            momentum=apcnetCfg['momentum'],
            weight_decay=apcnetCfg['weight_decay'])

        fcnheadCfg = self.config['optimizers']['FCNHead']
        self.scheduler['FCNHead'] = paddle.optimizer.lr.StepDecay(learning_rate=fcnheadCfg['lr'],
                                                                  step_size=self.optimizers_step_size, gamma=0.9,
                                                                  verbose=False)
        optimizers['FCNHead'] = paddle.optimizer.Momentum(
            parameters=self.models['FCNHead'].parameters(),
            learning_rate=self.scheduler['FCNHead'],
            momentum=fcnheadCfg['momentum'],
            weight_decay=fcnheadCfg['weight_decay'])
        return optimizers

    def init_criterions(self):
        criterions = {}
        criterions['celoss'] = paddle.nn.CrossEntropyLoss(ignore_index=255, reduction='mean', axis=1)
        return criterions
        pass

    def init_check(self):
        assert len(self.models) == 3
        assert len(self.optimizers) == 3
        assert len(self.criterions) == 1

    def train_step(self, batch_id, batch):
        self.models['backbone'].train()
        self.models['APCHead'].train()
        self.models['FCNHead'].train()

        self.optimizers['backbone'].clear_grad()
        self.optimizers['APCHead'].clear_grad()
        self.optimizers['FCNHead'].clear_grad()

        x, label = batch
        # print(x.shape)
        x = paddle.to_tensor(x, dtype='float32')
        label = paddle.to_tensor(label, dtype='int64')
        # print('label',label.shape)

        feature2, feature3 = self.models['backbone'](x)  # 0,1,2,3 fcnhead

        pre1 = self.models['APCHead'](feature3)
        pre1 = F.interpolate(x=pre1, size=[512, 1024], mode="bilinear", align_corners=False)
        loss_apc = 1.0 * self.criterions['celoss'](pre1, label)
        # loss_apc.backward(retain_graph=True)
        loss_apc_numpy = loss_apc.numpy()

        pre2 = self.models['FCNHead'](feature2)
        pre2 = F.interpolate(x=pre2, size=[512, 1024], mode="bilinear", align_corners=False)
        loss_fcn = 0.4 * self.criterions['celoss'](pre2, label)
        # loss_fcn.backward()
        loss_fcn_numpy = loss_fcn.numpy()

        loss = loss_apc + loss_fcn
        loss.backward()

        self.optimizers['backbone'].step()
        self.optimizers['APCHead'].step()
        self.optimizers['FCNHead'].step()

        self.scheduler['backbone'].step()
        self.scheduler['APCHead'].step()
        self.scheduler['FCNHead'].step()
        # del  x
        #
        # pre2 = self.models['FCNHead'](feature2)
        # pre2 = F.interpolate(x=pre2, size=[512, 1024], mode="bilinear", align_corners=False)
        # loss_fcn=0.4*self.criterions['celoss'](pre2,label)
        # loss=loss_fcn
        # loss_fcn_numpy=loss_fcn.numpy()
        # del feature2,pre2
        # # print('feature.shape',feature.shape)
        #
        # pre1=self.models['APCHead'](feature3)
        # pre1=F.interpolate(x=pre1, size=[512,1024],mode="bilinear",align_corners=False)
        # loss_apc=1.0 * self.criterions['celoss'](pre1, label)
        # loss_apc_numpy=loss_apc.numpy()
        # loss += loss_apc
        #
        #
        # del feature3,pre1,loss_apc

        # del label
        #
        # loss.backward()
        # self.optimizers['backbone'].step()
        # self.optimizers['APCHead'].step()
        # self.optimizers['FCNHead'].step()
        #
        # self.scheduler['APCHead'].step()
        # self.scheduler['FCNHead'].step()
        # if batch_id%self.optimizers_step_size==0:
        #     self.logger.info('adjust APCHead lr to {},FCNHead lr to {}'.format(self.scheduler['APCHead'].last_lr,self.scheduler['FCNHead'].last_lr))
        # self.logger.info('x type {} dtype {}'.format(type(x),x.dtype))
        # self.logger.info('label type {} dtype {}'.format(type(label),label.dtype))
        # self.logger.info('label {}'.format(label))

        # record=None
        # record='inter{} , loss:{}'.format(self.inter,self.LossTotal/self.train_display_step)
        # record='loss:{}'.format(loss.numpy())
        # del loss

        # del x,label,pre1,pre2,feature2,feature3,loss

        ##delete 12111 MiB
        ##delete all in end 12565 MiB
        ##not delete  12565 MiB
        return loss_apc_numpy, loss_fcn_numpy  # (apc,fcn)

    def val(self):
        self.models['backbone'].eval()
        self.models['APCHead'].eval()
        self.models['FCNHead'].eval()
        ignore_label255 = 255  # 0 ,255
        if not ignore_label255:
            confusionMatrix = np.zeros((20, 20))
        else:
            confusionMatrix = np.zeros((19, 19))
        for i, batch in enumerate(tqdm(self.dataloaders['val'])):
            x, label = batch
            x = paddle.to_tensor(x, dtype='float32')
            # print(label.shape)
            label = paddle.to_tensor(label, dtype='int64')
            # print('label',label.shape)
            feature2, feature3 = self.models['backbone'](x)  # 0,1,2,3
            # print('feature.shape',feature.shape)
            pre1 = self.models['APCHead'](feature3)
            pre2 = self.models['FCNHead'](feature2)
            pre1 = 1.0 * pre1 + 0.4 * pre2
            pre1 = F.interpolate(x=pre1, size=[512, 1024], mode="bilinear")  # ,align_corners=True
            # pre2=F.interpolate(x=pre2, size=[512,1024])
            prediction = paddle.argmax(pre1, axis=1).numpy()

            if not ignore_label255:
                prediction[np.where(label.numpy() == 255)] = 255
            else:
                pass
            confusionMatrix += self.getConfusionMatrix(prediction, label.numpy(), ignore_label=ignore_label255)
            del x, label, feature2, feature3, pre1, pre2, prediction
        miou, ious = self.getMiou(confusionMatrix, 19 if ignore_label255 else 20)
        self.logger.info('inter {} ,val miou:{}'.format(self.inter, miou))

    def save_checkpoint(self):
        state = {}
        state['inter'] = self.inter
        state['models'] = {}
        state['models']['backbone'] = self.models['backbone'].state_dict()
        state['models']['APCHead'] = self.models['APCHead'].state_dict()
        state['models']['FCNHead'] = self.models['FCNHead'].state_dict()
        state['optimizers'] = {}
        state['optimizers']['APCHead'] = self.optimizers['APCHead'].state_dict()
        state['optimizers']['FCNHead'] = self.optimizers['FCNHead'].state_dict()

        save_path = os.path.join(self.exp_dir, 'ckpt', str(self.inter) + '.pdparams')

        paddle.save(state, save_path)
        self.logger.info('save ckpt inter:{}'.format(self.inter))

    def delete_checkpoint(self):
        ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
        for ckpt in ckpts:
            inter = int(ckpt.split('/')[-1].split('.')[0])
            if inter != self.inter:
                delete_path = os.path.join(self.exp_dir, 'ckpt', str(inter) + '.pdparams')
                os.remove(delete_path)

    def load_checkpoint(self):
        if self.resume_inter == 'latest':

            ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
            latest = -1
            for ckpt in ckpts:
                inter = int(ckpt.split('/')[-1].split('.')[0])
                if inter > latest:
                    latest = inter
            load_path = os.path.join(self.exp_dir, 'ckpt', str(latest) + '.pdparams')
            pass
        else:
            load_path = os.path.join(self.exp_dir, 'ckpt', str(self.resume_inter) + '.pdparams')
        # print(load_path)
        try:
            state = paddle.load(load_path)
        except:
            if latest == -1:
                self.logger.info("no ckpt, no load ckpt")
                return
            os.remove(load_path)
            ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
            latest = -1
            for ckpt in ckpts:
                inter = int(ckpt.split('/')[-1].split('.')[0])
                if inter > latest:
                    latest = inter
            load_path = os.path.join(self.exp_dir, 'ckpt', str(latest) + '.pdparams')
            state = paddle.load(load_path)
        self.inter = state['inter'] + 1
        self.models['backbone'].set_state_dict(state['models']['backbone'])
        self.models['APCHead'].set_state_dict(state['models']['APCHead'])
        self.models['FCNHead'].set_state_dict(state['models']['FCNHead'])
        self.optimizers['APCHead'].set_state_dict(state['optimizers']['APCHead'])
        self.optimizers['FCNHead'].set_state_dict(state['optimizers']['FCNHead'])

        self.logger.info('resume ckpt from {}'.format(load_path))

    def getMiou(self, mat, labels_num=19):
        ious = []
        for i in range(labels_num):
            iou_i = mat[i][i] / (np.sum(mat[i], axis=0) + np.sum(mat[:, i], axis=0) - mat[i][i])
            ious.append(iou_i)

        res = (np.mean(ious), ious)
        return res
        # miou=np.trace(mat)/(2*np.sum(mat)-np.trace(mat))
        # return miou,1

    def getConfusionMatrix(self, prediction, target, ignore_label=255):
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


if __name__ == '__main__':
    # p=dict(a=1,b=2)
    # print(len(p))
    # print(len(p.items()))
    trainer = Trainer(expName='apcnet-cityscapes-test', resume=1, resume_inter='latest', config=CONFIG)
    trainer.run()
