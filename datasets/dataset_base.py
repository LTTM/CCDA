from os import path
import random

import cv2 as cv
import numpy as np

import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    # loads rgb & gt images from the provided lists & root paths
    # Performs data augmentation, rescaling & cropping
    # Maps the raw labels in the provided train-ids (overridable method)
    # Maps the train-id labels in step-ids (overridable method)
    def __init__(self, root_path, splits_path, resize_to=None, crop_size=None, split='train', list_path_separator=' ', augment=False, **kwargs):
        # initialize the self.items list [(im_path, gt_path)]
        self.load_items(root_path, splits_path, split, list_path_separator)
        # define class attributes used by mapping methods
        self.raw_to_train = None # if None ids are left as-is
        self.train_to_incremental = None # if None we are not in an incremental setup
        self.step = 0 # ignored if train_to_incremental is None
        self.cmap = np.array([[i,i,i] for i in range(256)], dtype=np.uint8) # default colormap, maps labels in grayscale images
        # save crop and resize dimensions
        self.resize_to = resize_to # if None, no rescaling is applied
        self.crop_size = crop_size # if None, no cropping is applied
        # save augment related attributes
        self.augment = augment
        self.kwargs = kwargs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        im_path, gt_path = self.items[item]
        im, gt = self.read_im(im_path), self.read_gt(gt_path)
        if self.augment:
            im, gt = self.data_augment(im, gt)
        im, gt = self.resize_and_crop(im, gt)
        gt = self.map_to_train(gt)
        gt = self.map_to_incremental(gt)
        return self.to_pytorch(im, gt)

    def load_items(self, root_path, splits_path, split, list_path_separator):
        # extract the actual path to the list
        lp = path.join(splits_path, split+'.txt')
        # populate the list by reading the file
        with open(lp, 'r') as f:
            tmp = [line.rstrip('\n').split(list_path_separator) for line in f]
        # strip an eventual leading '/' and join the rgb & gt path to the root path
        # noinspection PyAttributeOutsideInit
        self.items = list(map(lambda tup: tuple(map(lambda p: path.join(root_path, p.lstrip('/')), tup)), tmp))

    @staticmethod
    def to_pytorch(bgr, gt):
        bgr = np.transpose(bgr.astype(np.float32)-[104.00698793, 116.66876762, 122.6789143], (2, 0, 1))
        return torch.from_numpy(bgr), torch.from_numpy(gt)

    @staticmethod
    def to_rgb(tensor):
        t = np.array(tensor.transpose(0,1).transpose(1,2))+[104.00698793, 116.66876762, 122.6789143] # bgr
        t = np.round(t[...,::-1]).astype(np.uint8) # rgb
        return t

    @staticmethod
    def read_im(im_path):
        # image is read in bgr
        return cv.imread(im_path)

    @staticmethod
    def read_gt(im_path):
        # image should be grayscale
        return cv.imread(im_path, cv.IMREAD_UNCHANGED)

    def map_to_train(self, gt):
        clone = -np.ones(gt.shape, dtype=int)
        if self.raw_to_train is not None:
            for k,v in self.raw_to_train.items():
                clone[gt==k] = v
        return clone

    def map_to_incremental(self, gt):
        if self.train_to_incremental is not None:
            if self.ignored_indices is not None:
                if self.train_to_incremental[self.step] is not None:
                    clone = -np.ones(gt.shape, dtype=int)
                    for k,v in self.train_to_incremental[self.step].items():
                        clone[gt==k] = v if v not in self.ignored_indices[self.step] else -1
                    return clone
                else:
                    for k in self.ignored_indices[self.step]:
                        gt[gt==k] = -1
                    return gt
            else:
                if self.train_to_incremental[self.step] is not None:
                    clone = -np.ones(gt.shape, dtype=int)
                    for k,v in self.train_to_incremental[self.step].items():
                        clone[gt==k] = v
                    return clone
        return gt
    # noinspection PyArgumentList
    def resize_and_crop(self, rgb, gt):
        if self.resize_to is not None:
            rgb = cv.resize(rgb, self.resize_to, interpolation=cv.INTER_AREA) # usually images are downsized, best results obtained with inter_area
            gt = cv.resize(gt, self.resize_to, interpolation=cv.INTER_NEAREST_EXACT) # labels must be mapped as-is
        if self.crop_size is not None:
            H, W, _ = rgb.shape
            dh, dw = H-self.crop_size[1], W-self.crop_size[0]
            assert dh>=0 and dw >= 0, "Incompatible crop size: (%d, %d), images have dimensions: (%d, %d)"%(self.crop_size[0], self.crop_size[1], W, H)
            h0, w0 = random.randint(0, dh) if dh>0 else 0, random.randint(0, dw) if dw>0 else 0
            rgb = (rgb[h0:h0+self.crop_size[1], w0:w0+self.crop_size[0], ...]).copy()
            gt = (gt[h0:h0+self.crop_size[1], w0:w0+self.crop_size[0], ...]).copy()
        return rgb, gt.astype(np.long)

    def data_augment(self, im, gt):
        if self.kwargs['flip'] and random.random()<.5:
            im = (im[:,::-1,...]).copy()
            gt = (gt[:,::-1,...]).copy()
        if self.kwargs['gaussian_blur']:
            im = cv.GaussianBlur(im, (0,0), random.random()*self.kwargs['blur_mul'])
        return im, gt

    def get_cmap(self):
        if self.train_to_incremental is not None:
            return self.cmap[self.step]
        return self.cmap

    def color_label(self, gt):
        cur_cmap = self.get_cmap()
        return cur_cmap[np.array(gt)]
