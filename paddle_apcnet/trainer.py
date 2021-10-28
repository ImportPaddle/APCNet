
from tqdm import tqdm
import paddle
import datetime
import logging
import os
from dataloader import CityScapesDataset
from architectures import getApcNet
from paddle.io import Dataset,DataLoader
from  config import CONFIG
import paddle.nn.functional as F
import warnings
import glob 
warnings.filterwarnings('ignore')
class Trainer():
    def __init__(self,expName,resume=True,resume_inter='latest',config=None):
        super(Trainer,self).__init__()
        self.config=CONFIG
        self.root=os.getcwd()
        self.resume=resume
        self.resume_inter=resume_inter
        self.exp_dir=self.init_exp_dir(expName=expName)
        
        self.logger=self.init_logger()
        self.inter=0
        self.max_inter=65000
        self.models=self.init_models()
        self.dataloaders=self.init_dataloaders()
        self.optimizers=self.init_optimizers()
        self.criterions=self.init_criterions()
        self.init_check()
        self.stepEachEpoch=(len(self.dataloaders['train'])+1)
        self.train_display_step=50
        self.train_save_step=2000
        self.evl_step=3000
        self.logger.info('init complete')
    def run(self):
        ##resume
        if self.resume:
            self.resume_experiment()
            
        while(self.inter<self.max_inter):
            for batch_id, batch in enumerate(tqdm(self.dataloaders['train']())):
                record=self.train_step(batch_id,batch)
                if batch_id %  self.train_display_step == 0 or record:
                    self.logger.info(record)
                if self.inter%self.train_save_step==0:
                    self.save_checkpoint()
                    self.delete_checkpoint()
                self.inter+=1
                
    def resume_experiment(self):
        self.load_checkpoint()
        
    def init_exp_dir(self,expName):
        expdir=os.path.join(self.root,'experiments',expName)
        if os.path.exists(expdir) and not self.resume:
            # now_str = datetime.datetime.now().__str__().replace(' ','_')
            # expdir=os.path.join(self.root,'experiments',expName+'_'+now_str)
            # if os.path.exists(expdir):
            print('exist exp dir {}'.format(expdir))
            exit(1)
            # else:
            #     os.makedirs(expdir)
        else:
            
            os.makedirs(expdir,exist_ok=True)
            models_dir=os.path.join(expdir,'ckpt')
            os.makedirs(models_dir,exist_ok=True)
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
        return networks ##['backbone','APCHead','FCNHead']
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
        self.optimizers_step_size=int(self.max_inter/250)
        optimizers={}
        self.scheduler={}
        backboneCfg=self.config['optimizers']['backbone']
        optimizers['backbone']= paddle.optimizer.Momentum(
                                    parameters=self.models['backbone'].parameters(),
                                    learning_rate=backboneCfg['lr'],
                                    momentum=backboneCfg['momentum'],
                                    weight_decay=backboneCfg['weight_decay'])
                                                # paddle.optimizer.SGD(learning_rate=0.001, parameters=None, weight_decay=None)
        apcnetCfg=self.config['optimizers']['APCHead']
        
        # step_size=5
        self.scheduler['APCHead'] = paddle.optimizer.lr.StepDecay(learning_rate=apcnetCfg['lr'], step_size=1, gamma=0.98, verbose=False)
        optimizers['APCHead']= paddle.optimizer.Momentum(
                                    parameters=self.models['APCHead'].parameters(),
                                    learning_rate=self.scheduler['APCHead'],
                                    momentum=apcnetCfg['momentum'],
                                    weight_decay=apcnetCfg['weight_decay'])
        
        fcnheadCfg=self.config['optimizers']['FCNHead']
        self.scheduler['FCNHead'] = paddle.optimizer.lr.StepDecay(learning_rate=fcnheadCfg['lr'], step_size=1, gamma=0.98, verbose=False)
        optimizers['FCNHead']= paddle.optimizer.Momentum(
                                    parameters=self.models['FCNHead'].parameters(),
                                    learning_rate=self.scheduler['FCNHead'],
                                    momentum=fcnheadCfg['momentum'],
                                    weight_decay=fcnheadCfg['weight_decay'])
        return optimizers
    def init_criterions(self):
        criterions={}
        criterions['celoss']=paddle.nn.CrossEntropyLoss(ignore_index=255,reduction='mean',axis=1)
        return criterions
        pass
    def init_check(self):
        assert len(self.models)==3
        assert len(self.optimizers)==3
        assert len(self.criterions)==1
    def train_step(self,batch_id,batch):
        self.optimizers['backbone'].clear_grad()
        self.optimizers['APCHead'].clear_grad()
        x,label=batch
        x=paddle.to_tensor(x,dtype='float32')
        label=paddle.to_tensor(label,dtype='int64')
        # print('label',label.shape)
        feature2=self.models['backbone'](x)[0] #0,1,2,3
        feature3=self.models['backbone'](x)[1] #0,1,2,3
        # print('feature.shape',feature.shape)
        pre1=self.models['APCHead'](feature3)
        pre2=self.models['FCNHead'](feature2)
        pre1=F.interpolate(x=pre1, size=[512,1024])
        pre2=F.interpolate(x=pre1, size=[512,1024])
        # print('pre1.shape',pre1.shape)
        # print('pre2.shape',pre2.shape)
        """
        pre1.shape [1, 19, 64, 128]
        pre2.shape [1, 19, 64, 128]
        """
        # print(label.dtype)
        loss1=0.99*self.criterions['celoss'](pre1,label)+0.01*self.criterions['celoss'](pre2,label)
        
        loss1.backward()
        self.optimizers['backbone'].step()
        self.optimizers['APCHead'].step()
        self.optimizers['FCNHead'].step()
        
        
        if batch_id%self.optimizers_step_size==0:
            self.scheduler['APCHead'].step()
            self.scheduler['FCNHead'].step()
            self.logger.info('adjust APCHead lr to {},FCNHead lr to {}'.format(self.scheduler['APCHead'].last_lr,self.scheduler['FCNHead'].last_lr))
        # self.logger.info('x type {} dtype {}'.format(type(x),x.dtype))
        # self.logger.info('label type {} dtype {}'.format(type(label),label.dtype))
        # self.logger.info('label {}'.format(label))
        record=None
        return record
    def val(self):
        pass
    def save_checkpoint(self):
        state={}
        state['inter']=self.inter
        state['models']={}
        state['models']['backbone']=self.models['backbone'].state_dict()
        state['models']['APCHead']=self.models['APCHead'].state_dict()
        state['models']['FCNHead']=self.models['FCNHead'].state_dict()
        state['optimizers']={}
        state['optimizers']['APCHead']=self.optimizers['APCHead'].state_dict()
        state['optimizers']['FCNHead']=self.optimizers['FCNHead'].state_dict()
        
        save_path=os.path.join(self.exp_dir,'ckpt',str(self.inter)+'.pdparams')
        
        paddle.save(state,save_path)
        self.logger.info('save ckpt inter:{}'.format(self.inter))
    def delete_checkpoint(self):
        if self.inter!=0:
            
            delete_path=os.path.join(self.exp_dir,'ckpt',str(self.inter-self.train_save_step)+'.pdparams')
            os.remove(delete_path)
    def load_checkpoint(self):
        if self.resume_inter=='latest':
            ckpts=list(glob.glob(self.exp_dir+'/ckpt/*.pdparams'))
            latest=-1
            for ckpt in ckpts:
                inter=int(ckpt.split('/')[-1].split('.')[0])
                if inter>latest:
                    latest=inter
            load_path=os.path.join(self.exp_dir,'ckpt',str(latest)+'.pdparams')
            pass
        else:
            load_path=os.path.join(self.exp_dir,'ckpt',str(self.resume_inter)+'.pdparams')
        print(load_path)
        try:
            state=paddle.load(load_path)
        except:
            os.remove(load_path)
            ckpts=list(glob.glob(self.exp_dir+'/ckpt/*.pdparams'))
            latest=-1
            for ckpt in ckpts:
                inter=int(ckpt.split('/')[-1].split('.')[0])
                if inter>latest:
                    latest=inter
            load_path=os.path.join(self.exp_dir,'ckpt',str(latest)+'.pdparams')
            state=paddle.load(load_path)
        self.inter=state['inter']+1
        self.models['backbone'].set_state_dict(state['models']['backbone'])
        self.models['APCHead'].set_state_dict(state['models']['APCHead'])
        self.models['FCNHead'].set_state_dict(state['models']['FCNHead'])
        self.optimizers['APCHead'].set_state_dict(state['optimizers']['APCHead'])
        self.optimizers['FCNHead'].set_state_dict(state['optimizers']['FCNHead'])
        
        self.logger.info('resume ckpt from {}'.format(load_path))
if __name__=='__main__':
    # p=dict(a=1,b=2)
    # print(len(p))
    # print(len(p.items()))
    trainer=Trainer(expName='apcnet-cityscapes',resume=1,resume_inter='latest',config=CONFIG)
    trainer.run()