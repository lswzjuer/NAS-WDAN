import logging
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import ASGD
from torch.optim import Adamax
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import RMSprop


logger = logging.getLogger('optimizer of model')

key2opt =  {'sgd': SGD,
            'adam': Adam,
            'asgd': ASGD,
            'adamax': Adamax,
            'adadelta': Adadelta,
            'adagrad': Adagrad,
            'rmsprop': RMSprop,}

def get_optimizer(optimizer_name=None):
    if optimizer_name is None:
        logger.info("Using SGD optimizer")
        return SGD
    else:
        if optimizer_name not in key2opt:
            raise NotImplementedError('Optimizer {} not implemented'.format(optimizer_name))
        logger.info('Using {} optimizer'.format(optimizer_name))
        return key2opt[optimizer_name]
