import argparse
import os
import time
import shutil
from tensorboardX import SummaryWriter


import torch
torch.backends.cudnn.benchmark = True 
from torch.nn import CrossEntropyLoss
from torch.utils import data

from datasets.gta5_idd import GTAVIDDDataset
from datasets.idd import IDDDataset
from datasets.cityscapes import CityscapesDataset
from datasets.gta5 import GTAVDataset
from models.model import SegmentationModel
from utils.losses import *
from utils.misc import *
from utils.model import *
import logging
import json

class Initializer():
    def __init__(self):
        self.args = self.cliArguments()
        assert self.args.uda_skip_iterations < self.args.iterations, "Cannot skip more iterations than the total number"
        set_seed(self.args.seed) # set the seed

        self.initialize_supervised()
        self.initialize_incremental_losses()
        self.initialize_uda()
        self.initialize_logger()

    def cliArguments(self):
        argparser = argparse.ArgumentParser()

        self.datasetArguments(argparser)
        self.modelArguments(argparser)
        self.hyperparametersArguments(argparser)
        self.loggerArguments(argparser)

        return argparser.parse_args()

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
        argparser.add_argument('--dataloader_workers', default=4, type=int,
                               help='Number of workers to use for each dataloader (significantly affects RAM consumption)')

        argparser.add_argument('--blur_mul', default=1, type=int)

        argparser.add_argument('--pin_memory', default=False, type=str2bool)
        argparser.add_argument('--persistent_workers', default=False, type=str2bool)


        # Source Dataset Arguments
        argparser.add_argument('--source_dataset', default="gta", type=str,
                               choices=["gta",'cityscapes', "IDD"],
                               help='The dataset used for supervised training')
        argparser.add_argument('--source_rescale_size', default=[1280,720], type=str2intlist,
                               help='Size the images will be resized to during loading, before crop - syntax:"1280,720"')
        argparser.add_argument('--source_crop_images', default=False, type=str2bool,
                               help='Whether to crop the images or not')
        argparser.add_argument('--source_crop_size', default=[512,512], type=str2intlist,
                               help='Size the images will be cropped to - syntax:"1280,720"')
        argparser.add_argument('--source_root_path', type=str, help='Path to the dataset root folder')
        argparser.add_argument('--source_splits_path', type=str, help='Path to the dataset split lists')
        argparser.add_argument('--source_train_split', default='train', type=str,
                               help='Split file to be used for training samples')
        argparser.add_argument('--source_val_split', default='val', type=str,
                               help='Split file to be used for validation samples')

        # Target Dataset Arguments
        argparser.add_argument('--target_dataset', default='cityscapes', type=str2str_none_num,
                               choices=["gta",'cityscapes', "IDD", None],
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
        argparser.add_argument('--target_val_split', default='val', type=str,
                               help='Split file to be used for validation samples')

    @staticmethod
    def modelArguments(argparser):
        # Classifier Arguments
        argparser.add_argument('--classifier', default='DeepLabV3', type=str,
                               choices=['DeepLabV2', 'DeepLabV3', 'DeepLabV2MSIW', 'FCN', 'PSPNet'],
                               help='Which classifier head to use in the model')
        argparser.add_argument('--backbone', default='ResNet101', type=str,
                               choices=['ResNet101', 'ResNet50', 'VGG16', 'VGG13'],
                               help='Which backbone to use in the model')
        argparser.add_argument('--c2f_weight_init', default=True, type=str2bool,
                               help='Whether to use c2f weight initialization strategy')

        # load step 0 checkpoint
        argparser.add_argument('--step_model_checkpoint', default=None, type=str2str_none_num, 
                                help='Path to the pretrained model used for step0')

        argparser.add_argument('--continue_from_best_model', default=False, type=str2bool,
                                help="Wheter to start the next step from the best model fuond during validation of the previous step")


    @staticmethod
    def hyperparametersArguments(argparser):
        argparser.add_argument('--seed', default=12345, type=int,
                               help='Seed for the RNGs, for repeatability')
        argparser.add_argument('--lr', default=2.5e-4, type=float,
                               help='The learning rate to be used in the non-incremental steps')
        argparser.add_argument('--incremental_lr', default=1e-4, type=float,
                               help='The learning rate to be used in the incremental steps')
        argparser.add_argument('--poly_power', default=.9, type=float,
                               help='lr polynomial decay rate')
        argparser.add_argument('--decay_over_iterations', default=250000, type=int,
                               help='lr polynomial decay max_steps, count restarts after each incremental step')
        argparser.add_argument('--iterations', default=50000, type=int,
                               help='Number of iterations performed at step 0')
        argparser.add_argument('--incremental_iterations', default=25000, type=int,
                               help='Number of iterations performed at the incremental steps')
        argparser.add_argument('--uda_skip_iterations', default=25000, type=int,
                               help='Number of iterations (of step 0) to be skipped before starting to adapt the model')
        argparser.add_argument('--momentum', default=.9, type=float,
                               help='SGD optimizer momentum')
        argparser.add_argument('--weight_decay', default=1e-4, type=float,
                               help='SGD optimizer weight decay')
        argparser.add_argument('--kd_lambda', default=10., type=float,
                               help='Knowledge Distillation loss coefficient')
        argparser.add_argument('--kd_type', default="kdc", type=str,
                               choices=["kdc","mib", "distance"],
                               help='Knowledge Distillation type')
        argparser.add_argument('--kd_lambda_c2f', default=10, type=float,
                               help='hen different from kd_lambda, coarse to fine kd lambda')
        argparser.add_argument('--kd_distance', default='L2', type=str,
                               choices=['L1', 'L2'],
                               help='Which distance to use in the distance-based kd loss')
        argparser.add_argument('--kd_distance_on_logits', default=False, type=str2bool,
                               help='Whether to apply the distance-based kd loss after the softmax or not')
        argparser.add_argument('--msiw_lambda', default=.1, type=float,
                               help='MaxSquare loss coefficient')
        argparser.add_argument('--msiw_ratio', default=.2, type=float,
                               help='MaxSquare loss inference window ratio')

    @staticmethod
    def loggerArguments(argparser):
        argparser.add_argument('--validate_every_steps', default=2500, type=int,
                       help='Number of iterations every which a validation is run, <= 0 disables validation')
        argparser.add_argument('--logdir', default="log/%d"%(int(time.time())), type=str,
                       help='Path to the log directory')
        argparser.add_argument('--override_logs', default=False, type=str2bool,
                       help='Whether to override the logfolder')

    def initialize_supervised(self):
        # Supervised Training

        if self.args.source_dataset == "gta" and self.args.target_dataset == "IDD":
            sclass = GTAVIDDDataset
        elif self.args.source_dataset == "gta" and self.args.target_dataset == "cityscapes":
            sclass = GTAVDataset
        elif self.args.source_dataset == "cityscapes":
            sclass = CityscapesDataset
        elif self.args.source_dataset == "IDD":
            sclass = IDDDataset

        self.source_train = sclass(root_path=self.args.source_root_path,
                                   splits_path=self.args.source_splits_path,
                                   incremental_setup=self.args.incremental_setup,
                                   overlapped=self.args.overlapped,
                                   resize_to=self.args.source_rescale_size,
                                   crop_size=self.args.source_crop_size if self.args.source_crop_images else None,
                                   split=self.args.source_train_split,
                                   augment=self.args.augment_data,
                                   gaussian_blur=self.args.gaussian_blur,
                                   flip=self.args.gaussian_blur,
                                   blur_mul=self.args.blur_mul,
                                   )
        self.source_val = sclass(root_path=self.args.source_root_path,
                                 incremental_setup=self.args.incremental_setup,
                                 overlapped=False,
                                 splits_path=self.args.source_splits_path,
                                 resize_to=self.args.source_rescale_size,
                                 crop_size=None,
                                 split=self.args.source_val_split,
                                 augment=False,
                                 )
        self.sloader_train = data.DataLoader(self.source_train,
                                            shuffle=True,
                                            num_workers=self.args.dataloader_workers,
                                            batch_size=self.args.batch_size,
                                            drop_last=True,
                                            pin_memory=self.args.pin_memory,
                                            persistent_workers=self.args.persistent_workers,
                                            )
        self.sloader_val = data.DataLoader(self.source_val,
                                            shuffle=False,
                                            num_workers=self.args.dataloader_workers,
                                            batch_size=1,
                                            drop_last=False,
                                            pin_memory=self.args.pin_memory,
                                            persistent_workers=self.args.persistent_workers,
                                            )
        self.num_classes = len(self.source_train.cmap[self.source_train.step])-1 if self.args.incremental_setup \
                            else len(self.source_train.cmap)-1
        self.model = SegmentationModel(self.num_classes, self.args.backbone, self.args.classifier, pretrained=True)
        self.model.to("cuda")
        self.initialize_optimizer()
        self.ce_loss = CrossEntropyLoss(ignore_index=-1)
        self.ce_loss.to("cuda")

    def initialize_incremental_losses(self):
        # Incremental Learning
        if self.args.incremental_setup:
            if self.args.kd_lambda>0:
                if self.args.kd_type == "kdc": # KDC
                    self.kd_loss = c2f_kdc(ids_mapping=self.source_train.incremental_ids_mapping[self.source_train.step], same_kd_lambda=(self.args.kd_lambda_c2f==self.args.kd_lambda))
                    self.kd_loss.to("cuda")
                elif self.args.kd_type == "mib": # Original MiB using c2f_kdc
                    standard_mapping = self.source_train.incremental_ids_mapping[self.source_train.step]
                    background_ids = [b for a in standard_mapping if len(a)>1 for b in a]
                    mib_mapping = [a if len(a)==1 else background_ids for a in standard_mapping]
                    self.kd_loss = c2f_kdc(ids_mapping=mib_mapping)
                    self.kd_loss.to("cuda")
                elif self.args.kd_type == "distance": # L1/L2 KD
                    self.kd_loss = c2f_distance_kd(ids_mapping=self.source_train.incremental_ids_mapping[self.source_train.step],
                                                use_logits=self.args.kd_distance_on_logits,
                                                metric=self.args.kd_distance)
                    self.kd_loss.to("cuda")
                else:
                    self.kd_loss = None

    def initialize_uda(self):
        # Unsupervised Domain Adaptation
        self.uda = self.args.target_dataset is not None
        self.use_msiw_loss = self.args.msiw_lambda > 0 and self.uda

        if self.uda:
            if self.args.target_dataset == "gta":
                tclass = GTAVDataset
            elif self.args.target_dataset == "cityscapes":
                tclass = CityscapesDataset
            elif self.args.target_dataset == "IDD":
                tclass = IDDDataset
            
            self.target_train = tclass(root_path=self.args.target_root_path,
                                       splits_path=self.args.target_splits_path,
                                       incremental_setup=self.args.incremental_setup,
                                       overlapped=self.args.overlapped,
                                       resize_to=self.args.target_rescale_size,
                                       crop_size=self.args.target_crop_size if self.args.target_crop_images else None,
                                       split=self.args.target_train_split,
                                       augment=self.args.augment_data,
                                       gaussian_blur=self.args.gaussian_blur,
                                       flip=self.args.gaussian_blur,
                                        blur_mul=self.args.blur_mul,
                                        )
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
            
            # at step 0 i will use msiw as uda loss
            self.msiw_loss = MSIW(ratio=self.args.msiw_ratio)
            self.msiw_loss.to("cuda")

            self.mIoU_list_per_step = []

    def initialize_logger(self):
        if os.path.exists(self.args.logdir):
            # delete the logfolder, if present
            if self.args.override_logs:
                shutil.rmtree(self.args.logdir)
                
            else:
                raise ValueError("Logdir '%s' already exists, cannot proceed. \
                                 If you wish to override it enable the override_logs CLI flag."%self.args.logdir)

        # create the log path
        os.makedirs(self.args.logdir, exist_ok=True)
        self.writer = SummaryWriter(self.args.logdir, flush_secs=10)

        # logger configure
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.args.logdir, 'train_log.txt'))
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
        set_seed(self.args.seed*(self.source_train.step+2))
        self.old_model = self.model
        self.old_model.eval()
        self.model = initialize_from_model(self.old_model,
                                           self.source_train.incremental_ids_mapping[self.source_train.step],
                                           self.args.c2f_weight_init)
        self.initialize_optimizer()
        self.initialize_incremental_losses()
        self.source_train.step += 1
        self.source_val.step += 1
        if self.uda:
            self.target_val.step += 1
            self.target_train.step += 1
        self.num_classes = len(self.source_train.cmap[self.source_train.step])-1 if self.args.incremental_setup \
                            else len(self.source_train.cmap)-1
        self.mIoU_list_per_step = []

    def initialize_i_step_model(self, step=None, test=False):
        try:
            if test and not self.args.continue_from_best_model:
                # find the next model to be loaded (step), used in test.py lo load sequentially all the models
                model_filename = "step".join(self.args.step_model_checkpoint.split("step")[:-1])+"step"+str(step)+".pth"
            elif test and self.args.continue_from_best_model:
                model_filename = "step".join(self.args.step_model_checkpoint.split("step")[:-1])+"step"+str(step)+"_best.pth"
            else:
                model_filename = self.args.step_model_checkpoint

            print("Loading checkpoint {}".format(model_filename))

            checkpoint = torch.load(model_filename)      
            self.model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully from {}".format(model_filename))

        except OSError:
            print("No checkpoint exists from {}..".format(model_filename))
            exit()

    def initialize_optimizer(self):
        self.optimizer = torch.optim.SGD(params=self.model.parameters_dict,
                                 momentum=self.args.momentum,
                                 weight_decay=self.args.weight_decay)

    def lr_scheduler(self, iteration, incremental=False):
        init_lr = self.args.incremental_lr if incremental else self.args.lr
        max_iters = self.args.decay_over_iterations/self.args.batch_size
        lr = init_lr * (1 - float(iteration) / max_iters) ** self.args.poly_power
        self.optimizer.param_groups[0]["lr"] = lr
        if len(self.optimizer.param_groups) == 2:
            self.optimizer.param_groups[1]["lr"] = 10 * lr
        return lr
