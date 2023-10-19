# TODO 改为输入

class Config(object):
    def __init__(self):
        # data
        self.expression_normalize = False
        self.image_normalize = False

        # model
        self.pretrained = True
        self.epoch = 1
        self.batch_size = 32
        self.encoder_lr = 1e-5
        self.header_lr = 1e-3
        self.model_name = 'resnet18'
        assert self.model_name in ['vgg16', 'densenet121', 'resnet18', 'resnet50']

        # train
        self.train_switch = False

        # log


config = Config()
