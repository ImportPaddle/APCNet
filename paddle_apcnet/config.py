CONFIG={}
CONFIG['train_batch_size']=2
CONFIG['val_batch_size']=1
CONFIG['test_batch_size']=1
CONFIG['num_workers'] =4 
CONFIG['optimizers']={
    'backbone':dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005),
    'APCHead':dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0005),
     'FCNHead':dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0005),
    }
# config['network']={'backbone':'resnet101','apc':}