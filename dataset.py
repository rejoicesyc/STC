import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import cv2
import torch
from torchvision import transforms

from scipy.ndimage.interpolation import rotate
import torch.nn.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt, mask, edge, grays):
        # assert img.size == mask.size
        if img.size == mask.size:
            pass
        else:
            print(img.size, mask.size)

        for t in self.transforms:
            img, gt, mask, edge, grays = t(img, gt, mask, edge, grays)
        return img, gt, mask, edge, grays


class RandomHorizontallyFlip(object):
    def __call__(self, img, gt, mask, edge, grays):
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT) , mask.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(
                Image.FLIP_LEFT_RIGHT), grays.transpose(Image.FLIP_LEFT_RIGHT)
        return img, gt, mask, edge, grays


class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    def __call__(self, img, mask):
        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)
        return img, mask


class RandomRotate(object):
    def __call__(self, img, mask, edge, angle_range=(0, 180)):
        self.degree = np.random.randint(*angle_range)
        rotate_degree = np.random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), edge.rotate(
            rotate_degree, Image.NEAREST)


class RandomScaleCrop(object):
    def __init__(self, input_size, scale_factor):

        self.input_size = input_size
        self.scale_factor = scale_factor

    def __call__(self, img, mask):
        # random scale (short edge)
        assert img.size[0] == self.input_size

        o_size = np.random.randint(int(self.input_size * 1), int(self.input_size * self.scale_factor))
        img = img.resize((o_size, o_size), resample=Image.BILINEAR)
        mask = mask.resize((o_size, o_size), resample=Image.NEAREST)  #

        # random crop input_size
        x1 = np.random.randint(0, o_size - self.input_size)
        y1 = np.random.randint(0, o_size - self.input_size)
        img = img.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))
        mask = mask.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))

        return img, mask


class ScaleCenterCrop(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, mask):
        w, h = img.size
        if w > h:
            oh = self.input_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.input_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), resample=Image.BILINEAR)
        mask = mask.resize((ow, oh), resample=Image.NEAREST)

        w, h = img.size
        x1 = int(round((w - self.input_size) / 2.0))
        y1 = int(round((h - self.input_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))
        mask = mask.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))

        return img, mask


class RandomGaussianBlur(object):
    def __call__(self, img, mask):
        if np.random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.random()))

        return img, mask


class RandomCrop(object):
    def __call__(self, image, gt, mask, edge, grays):
        image = np.array(image)
        gt = np.array(gt)
        mask = np.array(mask)
        edge = np.array(edge)
        grays = np.array(grays)
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        image = Image.fromarray(image[p0:p1, p2:p3, :])
        gt    = Image.fromarray(gt[p0:p1, p2:p3].astype('uint8'))
        mask = Image.fromarray(mask[p0:p1, p2:p3].astype('uint8'))
        edge = Image.fromarray(edge[p0:p1, p2:p3].astype('uint8'))
        grays = Image.fromarray(grays[p0:p1, p2:p3].astype('uint8'))

        return image, gt, mask, edge, grays


####################################################################

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

class UData(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.image_transform_train = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(448, 448)),
            transforms.Resize((352, 352)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.image_transform_test = transforms.Compose([
            transforms.Resize((352, 352)),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())
            print(f'dataset getting {len(self.samples)} samples')

    def __getitem__(self, idx):
        name = self.samples[idx]
        name = name.split('.')[0]
        image = Image.open(self.cfg.datapath + '/image/' + name + '.jpg').convert('RGB')

        if self.cfg.mode == 'train':
            image = self.image_transform_train(image)
            return image
        else:
            shape = image.size[::-1]
            image = self.image_transform_test(image)
            return image, shape, name

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cfg = Config(mode='train', datapath='./dataset/DUTS/DUTS-TR')
    data = UData(cfg)

    for image, mask, edge in data:
        image = np.array(image).transpose((1, 2, 0))
        mask = np.array(mask).squeeze()
        edge = np.array(edge).squeeze()

        print(image.shape, type(image))

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(image)  
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.subplot(1, 3, 3)
        plt.imshow(edge)
        plt.show()
        plt.pause(1)
        # input()
