import argparse
import os
import time
import shutil
from tensorboardX import SummaryWriter


import torch
torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.utils import data

from datasets.cityscapes import CityscapesDataset
from datasets.gta5 import GTAVDataset
from datasets.idd import IDDDataset

from models.model import SegmentationModel
from utils.losses import *
from utils.misc import *
from utils.model import *
import logging
from utils.train_initializer import Initializer

class TestInitializer(Initializer):
    def __init__(self):
        self.args = self.cliArguments()
        set_seed(self.args.seed) # set the seed
        self.initialize_logger()
        self.initialize_dataset()

    @staticmethod
    def datasetArguments(argparser):
        # Common Dataset Arguments
        argparser.add_argument('--incremental_setup', default='c2f', type=str2str_none_num,
                               choices=['c2f', None],
                               help='Type of incremental setup, none means no incremental steps')
        argparser.add_argument('--augment_data', default=True, type=str2bool,
                               help='Whether to augment the (training) images with flip & Gaussian Blur')
        argparser.add_argument('--random_flip', default=True, type=str2bool,
                               help='Whether to randomly flip images l/r')
        argparser.add_argument('--gaussian_blur', default=True, type=str2bool,
                               help='Whether to apply random gaussian blurring')
        argparser.add_argument('--batch_size', default=1, type=int,
                               help='Training batch size')
        argparser.add_argument('--overlapped', default=True, type=str2bool,
                               help='Whether to keep only the new fine classes in each step')
        argparser.add_argument('--dataloader_workers', default=3, type=int,
                               help='Number of workers to use for each dataloader (significantly affects RAM consumption)')

        argparser.add_argument('--pin_memory', default=False, type=str2bool)
        argparser.add_argument('--persistent_workers', default=False, type=str2bool)

        # Target Dataset Arguments
        argparser.add_argument('--target_dataset', default='cityscapes', type=str2str_none_num,
                               choices=['gta','cityscapes', "IDD", None],
                               help='The dataset we wish to adapt to, must be different from source_dataset. none means no adaptation.')
        argparser.add_argument('--target_rescale_size', default=[1280,640], type=str2intlist,
                               help='Size the images will be resized to during loading, before crop - syntax:"1280,720"')
        argparser.add_argument('--target_crop_images', default=False, type=str2bool,
                               help='Whether to crop the images or not')
        argparser.add_argument('--target_crop_size', default=[512,512], type=str2intlist,
                               help='Size the images will be cropped to - syntax:"1280,720"')
        argparser.add_argument('--target_root_path', type=str, help='Path to the dataset root folder')
        argparser.add_argument('--target_splits_path', type=str, help='Path to the dataset split lists')
        argparser.add_argument('--target_train_split', default='train', type=str,
                               help='Split file to be used for training samples')
        argparser.add_argument('--target_val_split', default='test', type=str,
                               help='Split file to be used for testing samples')


    @staticmethod
    def hyperparametersArguments(argparser):
        argparser.add_argument('--seed', default=12345, type=int,
                               help='Seed for the RNGs, for repeatability')
        argparser.add_argument('--momentum', default=.9, type=float,
                               help='SGD optimizer momentum')
        argparser.add_argument('--weight_decay', default=1e-4, type=float,
                               help='SGD optimizer weight decay')

    @staticmethod
    def loggerArguments(argparser):
        argparser.add_argument('--logdir', default="log/%d"%(int(time.time())), type=str,
                       help='Path to the log directory')
        
        argparser.add_argument('--load_MSIW_model', default=False, type=str2bool,
                                help='whether to use or not the model from MSIW (it has different names in the layers)')


    def initialize_dataset(self):

        if self.args.target_dataset == 'gta':
            tclass = GTAVDataset
        elif self.args.target_dataset == "cityscapes":
            tclass = CityscapesDataset
        elif self.args.target_dataset == "IDD":
            tclass = IDDDataset

        self.target_train = tclass(root_path=self.args.target_root_path,
                                    splits_path=self.args.target_splits_path,
                                    incremental_setup=self.args.incremental_setup,
                                    overlapped=True,
                                    resize_to=self.args.target_rescale_size,
                                    crop_size=self.args.target_crop_size if self.args.target_crop_images else None,
                                    split=self.args.target_train_split,
                                    augment=self.args.augment_data,
                                    gaussian_blur=self.args.gaussian_blur,
                                    flip=self.args.gaussian_blur)
        self.target_val = tclass(root_path=self.args.target_root_path,
                                    incremental_setup=self.args.incremental_setup,
                                    overlapped=False,
                                    splits_path=self.args.target_splits_path,
                                    resize_to=self.args.target_rescale_size,
                                    crop_size=None,
                                    split=self.args.target_val_split,
                                    augment=False)
        self.tloader_train = data.DataLoader(self.target_train,
                                            shuffle=True,
                                            num_workers=self.args.dataloader_workers,
                                            batch_size=self.args.batch_size,
                                            drop_last=True,
                                            pin_memory=self.args.pin_memory,
                                            persistent_workers=self.args.persistent_workers,
                                            )
        self.tloader_val = data.DataLoader(self.target_val,
                                        shuffle=False,
                                        num_workers=self.args.dataloader_workers,
                                        batch_size=1,
                                        drop_last=False,
                                        pin_memory=self.args.pin_memory,
                                        persistent_workers=self.args.persistent_workers,
                                        )

        self.num_classes = len(self.target_train.cmap[self.target_train.step])-1 if self.args.incremental_setup \
                            else len(self.target_train.cmap)-1
        self.model = SegmentationModel(self.num_classes, self.args.backbone, self.args.classifier, pretrained=True)
        self.model.to('cuda')


    def initialize_logger(self):
        # create the log path
        os.makedirs(self.args.logdir, exist_ok=True)
        self.writer = SummaryWriter(self.args.logdir, flush_secs=10)

        # logger configure
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.args.logdir, 'test_log.txt'))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        self.logger = logger
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))

    def next_step(self):
        set_seed(self.args.seed*(self.target_train.step+2))
        print("new step seed: ",  self.args.seed*(self.target_train.step+2))
        self.old_model = self.model
        self.old_model.eval()
        self.model = initialize_from_model(self.old_model,
                                           self.target_train.incremental_ids_mapping[self.target_train.step],
                                           self.args.c2f_weight_init)
        self.target_val.step += 1
        self.target_train.step += 1
        self.num_classes = len(self.target_train.cmap[self.target_train.step])-1 if self.args.incremental_setup \
                            else len(self.target_train.cmap)-1

