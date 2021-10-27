CONFIG={}
CONFIG['train_batch_size']=8
CONFIG['val_batch_size']=4
CONFIG['test_batch_size']=1
CONFIG['num_workers'] =4 
CONFIG['optimizers']={
    'backbone':dict(type='SGD', lr=1e-5, momentum=0.9, weight_decay=0.0005),
    'APCHead':dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    }
# config['network']={'backbone':'resnet101','apc':}