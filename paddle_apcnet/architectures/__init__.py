from .apcnet_paddle import APCHead
from .resnet_paddle import resnet101
from .fcnhead_paddle import FCNHead
def getApcNet():
    models={}
    models['backbone'],msg1=resnet101()
    models['APCHead']=APCHead()
    models['FCNHead']=FCNHead()
    return models,msg1