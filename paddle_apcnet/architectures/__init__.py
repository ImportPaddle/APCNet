from .apcnet_paddle import APCHead
from .resnet_paddle import resnet101
def getApcNet():
    models={}
    models['backbone'],msg1=resnet101()
    models['APCHead']=APCHead()
    return models,msg1