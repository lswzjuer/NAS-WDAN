import torch
import numpy as np
import os
from collections import OrderedDict
import logging
import shutil
import subprocess
import sys

def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def get_logger(log_dir):
    create_exp_dir(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('Nas Seg')
    logger.addHandler(fh)
    return logger

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def get_gpus_memory_info():
    """Get the maximum free usage memory of gpu"""
    rst = subprocess.run('nvidia-smi -q -d Memory',stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    rst = rst.strip().split('\n')
    memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2]
    id = int(np.argmax(memory_available))
    return id, memory_available

def calc_parameters_count(model):
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6



if __name__=="__main__":
    get_gpus_memory_info()