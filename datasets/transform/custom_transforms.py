import torch
import random
import numpy as np
import torchvision.transforms.functional as tf
from torchvision import transforms as T
from PIL import Image, ImageOps, ImageFilter
import numbers
import cv2

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,mask):
        for t in self.transforms:
            img,mask = t(img,mask)
        return img,mask


class Totensor_Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.),cvc=False):
        self.mean = mean
        self.std = std
        self.cvc=cvc
    def __call__(self, img,mask):
        # h w c ---> c h w    h w --> 1 h w tensor  input is PIL image type
        assert img.size == mask.size
        # conver PIL.Image to tensor and value range is 0~1
        img=tf.to_tensor(img)
        if not self.cvc:
            mask=torch.from_numpy(np.asarray(mask).astype(np.float32)).unsqueeze(0)
        else:
            mask=tf.to_tensor(mask)
        # normalize
        #img=tf.normalize(img,mean=self.mean,std=self.std)
        return img,mask




########################### color shift #################################
class AdjustSaturation(object):
    def __init__(self, saturation=0.2):
        self.saturation = saturation
    def __call__(self, img,mask):
        assert img.size == mask.size
        img=tf.adjust_saturation(img,random.uniform(1 - self.saturation,1 + self.saturation))
        return img,mask

class AdjustHue(object):
    def __init__(self, hue=0.02):
        self.hue = hue
    def __call__(self, img,mask):
        assert img.size == mask.size
        img=tf.adjust_hue(img, random.uniform(-self.hue,self.hue))
        return img,mask

class AdjustBrightness(object):
    def __init__(self, bf=0.2):
        self.bf = bf
    def __call__(self, img,mask):
        assert img.size == mask.size
        img=tf.adjust_brightness(img,random.uniform(1 - self.bf,1 + self.bf))
        return img,mask

class AdjustContrast(object):
    def __init__(self, cf=0.2):
        self.cf = cf
    def __call__(self, img,mask):
        assert img.size == mask.size
        img=tf.adjust_contrast(img,random.uniform(1 - self.cf,1 + self.cf))
        return img,mask

class RandomGaussianBlur(object):
    def __call__(self, img,mask):
        assert img.size == mask.size
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img,mask

class RandomColorJitter(object):
    def __init__(self,brightness=0.2,contrast=0.2,hue=0.02):
        self.br=brightness
        self.co=contrast
        self.hue=hue
        self.color_transform=T.ColorJitter(brightness=self.br,contrast=self.co,hue=self.hue)
    def __call__(self, img,mask):
        assert img.size==mask.size
        if random.random() < 0.5:
            img=self.color_transform(img)
        return img,mask

############################# Morphological transform ###################################
class RandomHorizontalFlip(object):
    def __call__(self,img,mask):
        assert img.size==mask.size
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img,mask

class RandomVerticallyFlip(object):
    def __call__(self, img,mask):
        assert img.size == mask.size
        if random.random() < 0.5:
            img=img.transpose(Image.FLIP_TOP_BOTTOM)
            mask=mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img,mask

class RandomRotate(object):
    def __init__(self, degree=180):
        ''' rotate rate number'''
        self.degree = degree
    def __call__(self, img,mask):
        assert img.size == mask.size
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        return img,mask


class RandomZoom(object):
    def __init__(self, size=2):
        ''' h,w zoom times size'''
        assert isinstance(size,tuple) or isinstance(size,list),"the type(size) is wrong !"
        self.h_times=size[0]
        self.w_times=size[1]

    def __call__(self, img,mask):
        assert img.size == mask.size
        # w,h
        if random.random() < 0.5:
            new_size = (int(img.size[0]*self.w_times), int(img.size[1]*self.h_times))
            img=img.resize(new_size, Image.BILINEAR)
            mask=mask.resize(new_size, Image.NEAREST)
        return img,mask

class FixedResize(object):
    def __init__(self, size=(224,224)):
        ''' input is resize (h ,w )'''
        assert isinstance(size,tuple) or isinstance(size,list),"the type(size) is wrong !"
        self.h=size[0]
        self.w=size[1]
    def __call__(self, img,mask):
        assert img.size == mask.size
        img=img.resize((self.w,self.h), Image.BILINEAR)
        mask=mask.resize((self.w,self.h), Image.NEAREST)
        return img,mask

class RandomScaleCrop(object):
    def __init__(self, crop_size=256, base_size=300,fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img,mask):
        # random scale (short edge)
        assert img.size == mask.size
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img,mask


# fixed center crop
class FixScaleCrop(object):
    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, img,mask):
        assert img.size == mask.size
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img,mask


class Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img,mask):
        assert img.size == mask.size
        img=ImageOps.expand(img, border=self.padding, fill=self.fill)
        mask=ImageOps.expand(mask, border=self.padding, fill=self.fill)
        return img,mask




#################### Special transformation #################################
class RandomElasticTransform(object):
    def __init__(self, alpha = 1.5, sigma=0.07, img_type='L'):
        self.alpha = alpha
        self.sigma = sigma
        self.img_type = img_type

    def _elastic_transform(self, img, mask):
        # convert to numpy
        img = np.array(img)  # hxwxc
        mask = np.array(mask)
        shape1=img.shape
        alpha = self.alpha*shape1[0]
        sigma = self.sigma*shape1[0]

        x, y = np.meshgrid(np.arange(shape1[0]), np.arange(shape1[1]), indexing='ij')
        blur_size = int(4 * sigma) | 1
        dx = cv2.GaussianBlur((np.random.rand(shape1[0], shape1[1]) * 2 - 1), ksize=(blur_size, blur_size),sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(shape1[0], shape1[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

        if (x is None) or (y is None):
            x, y = np.meshgrid(np.arange(shape1[0]), np.arange(shape1[1]), indexing='ij')

        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        # convert map
        map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

        img = cv2.remap(img, map_y, map_x, interpolation=cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT).reshape(shape1)
        mask =  cv2.remap(mask, map_y, map_x, interpolation=cv2.INTER_NEAREST, borderMode = cv2.BORDER_CONSTANT).reshape(shape1)

        img=Image.fromarray(img,mode=self.img_type)
        mask=Image.fromarray(mask, mode='L')
        return img,mask

    def __call__(self, img,mask):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        assert img.size == mask.size
        if random.random() < 0.5:
            return self._elastic_transform(img, mask)
        else:
            return img,mask

