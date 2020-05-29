class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'voc':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset=='isic2018':
            return r'E:\datasets\isic2018'
        elif dataset=='cvc':
            return r'E:\datasets\CVC-ClinicDB'
        elif dataset=='refuge':
            return r'C:\Users\rileyliu\Desktop\REFUGE'
        elif dataset=='chaos':
            return r'C:\Users\rileyliu\Desktop\CHAO'
        elif dataset=="lits19":
            return r'F:\LITS19'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
