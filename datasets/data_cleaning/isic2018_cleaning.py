import os
import argparse
import random
import shutil
from shutil import copyfile



def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)


    filenames = os.listdir(config.origin_data_path)
    data_list = []
    GT_list = []

    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext =='.jpg':
            filename = filename.split('_')[-1][:-len('.jpg')]
            data_list.append('ISIC_'+filename+'.jpg')
            GT_list.append('ISIC_'+filename+'_segmentation.png')

    num_total = len(data_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio))*num_total)
    num_valid = num_total-num_train


    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)


    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()
        
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.train_path,data_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.train_GT_path, GT_list[idx])
        copyfile(src, dst)


    for i in range(num_valid):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.valid_path,data_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.valid_GT_path, GT_list[idx])
        copyfile(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.2)


    # data path
    parser.add_argument('--origin_data_path', type=str, default='/extracephonline/medai_data2/rileyliu/datasets/ISIC2018/Task1/ISIC2018_Task1-2_Training_Input')
    parser.add_argument('--origin_GT_path', type=str, default='/extracephonline/medai_data2/rileyliu/datasets/ISIC2018/Task1/ISIC2018_Task1_Training_GroundTruth')
    
    parser.add_argument('--train_path', type=str, default='/extracephonline/medai_data2/rileyliu/datasets/ISIC2018/Task1_apply/train')
    parser.add_argument('--train_GT_path', type=str, default='/extracephonline/medai_data2/rileyliu/datasets/ISIC2018/Task1_apply/train_GT')
    parser.add_argument('--valid_path', type=str, default='/extracephonline/medai_data2/rileyliu/datasets/ISIC2018/Task1_apply/valid')
    parser.add_argument('--valid_GT_path', type=str, default='/extracephonline/medai_data2/rileyliu/datasets/ISIC2018/Task1_apply/valid_GT')


    config = parser.parse_args()
    print(config)
    main(config)