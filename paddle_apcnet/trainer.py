
from tqdm import tqdm
import paddle
import datetime
import logging
import os
from dataloader import CityScapesDataset
from architectures import getApcNet
from paddle.io import Dataset,DataLoader
from  config import CONFIG
class Trainer():
    def __init__(self,expName,resume=True,config=None):
        super(Trainer,self).__init__()
        self.config=CONFIG
        self.root=os.getcwd()
        self.resume=resume
        self.exp_dir=self.init_exp_dir(expName=expName)
        self.logger=self.init_logger()
        self.inter=0
        self.max_inter=65000
        self.models=self.init_models()
        self.dataloaders=self.init_dataloaders()
        self.optimizers=self.init_optimizers()
        self.criterions=self.init_criterions()
        self.init_check()
        self.train_display_step=50
        self.evl_step=3000
        self.logger.info('init complete')
    def run(self):
        ##resume
        while(self.inter<self.max_inter):
            for batch_id, batch in enumerate(tqdm(self.dataloaders['train']())):
                record=self.train_step(batch)
                if batch_id %  self.train_display_step == 0:
                    self.logger.info(record) 
                self.inter+=1
    def init_exp_dir(self,expName):
        expdir=os.path.join(self.root,'experiments',expName)
        if os.path.exists(expdir):
            now_str = datetime.datetime.now().__str__().replace(' ','_')
            expdir=os.path.join(self.root,'experiments',expName+'_'+now_str)
            if os.path.exists(expdir):
                print('exist exp dir ')
                exit(1)
            else:
                os.makedirs(expdir)
        else:
            os.makedirs(expdir)
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
        now_str = datetime.datetime.now().__str__().replace(' ','_')
        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        logger.addHandler(self.log_fileHandler)
        return logger
    def init_models(self):
        networks,msg_resnet=getApcNet()
        self.logger.info(msg_resnet)
        return networks ##['backbone','APCHead']
        pass
    def init_dataloaders(self):
        # trainLoader,valLoader,testLoader
        trainset=CityScapesDataset(root='../dataset/cityscapes',mode='train',SEED=1)
        valset=CityScapesDataset(root='../dataset/cityscapes',mode='val',SEED=1)
        testset=CityScapesDataset(root='../dataset/cityscapes',mode='test',SEED=1)
        trainLoader=DataLoader(trainset,batch_size=self.config['train_batch_size'],drop_last=False, num_workers=self.config['num_workers'],shuffle=True,use_buffer_reader=True)
        valLoader=DataLoader(valset,batch_size=self.config['val_batch_size'],drop_last=False,num_workers=self.config['num_workers'], shuffle=True,use_buffer_reader=True)
        testLoader=DataLoader(testset,batch_size=self.config['test_batch_size'],drop_last=False,num_workers=self.config['num_workers'], shuffle=True,use_buffer_reader=True)
        print(len(trainset))
        print(len(valset))
        print(len(testset))
        
        assert len(trainset)+len(valset)+len(testset)==5000,'images number is not 5000'
        assert len(trainset)==2975
        assert len(valset)==500
        assert len(testset)==1525
        dataloaders={}
        dataloaders['train']=trainLoader
        dataloaders['val']=valLoader
        dataloaders['test']=testLoader
        return dataloaders
    def init_optimizers(self):
        optimizers={}
        backboneCfg=self.config['optimizers']['backbone']
        optimizers['backbone']= paddle.optimizer.Momentum(
                                    parameters=self.models['backbone'].parameters(),
                                    learning_rate=backboneCfg['lr'],
                                    momentum=backboneCfg['momentum'],
                                    weight_decay=backboneCfg['weight_decay'])
                                                # paddle.optimizer.SGD(learning_rate=0.001, parameters=None, weight_decay=None)
        apcnetCfg=self.config['optimizers']['APCHead']
        scheduler = paddle.optimizer.lr.StepDecay(learning_rate=apcnetCfg['lr'], step_size=5, gamma=0.8, verbose=True)
        optimizers['APCHead']= paddle.optimizer.Momentum(
                                    parameters=self.models['APCHead'].parameters(),
                                    learning_rate=scheduler,
                                    momentum=apcnetCfg['momentum'],
                                    weight_decay=apcnetCfg['weight_decay'])
        return optimizers
    def init_criterions(self):
        criterions={}
        criterions['celoss']=paddle.nn.CrossEntropyLoss()
        return criterions
        pass
    def init_check(self):
        assert len(self.models)==2
        assert len(self.optimizers)==2
        assert len(self.criterions)==1
    def train_step(self,batch):
        self.optimizers['backbone'].clear_grad()
        self.optimizers['APCHead'].clear_grad()
        x,label=batch
        feature=self.models['backbone'](x)
        pre=self.models['APCHead'](feature)
        # print('pre.shape',pre.shape)
        loss=self.criterions['celoss'](pre,label)
        
        loss.backward()
        self.optimizers['backbone'].step()
        self.optimizers['APCHead'].step()
        # self.logger.info('x type {} dtype {}'.format(type(x),x.dtype))
        # self.logger.info('label type {} dtype {}'.format(type(label),label.dtype))
        # self.logger.info('label {}'.format(label))
        record=''
        pass
        return record
    def val_step(self):
        pass
    def save_checkpoint(self, dir, suffix=''):
        pass
    def load_checkpoint(self, dir):
        pass

if __name__=='__main__':
    # p=dict(a=1,b=2)
    # print(len(p))
    # print(len(p.items()))
    trainer=Trainer(resume=False,expName='test',config=CONFIG)
    trainer.run()