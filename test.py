import numpy as np
import os
import torch
from tqdm import tqdm

from utils.test_initializer import TestInitializer

from utils.metrics import Metrics

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Tester(TestInitializer):
    def test(self):
        
        if "best" in self.args.step_model_checkpoint:
            step_checkpoint = int(self.args.step_model_checkpoint.split("step")[-1].split("_best")[0])
        else:
            # extract the step from the checkpoint model provided       
            step_checkpoint = int(self.args.step_model_checkpoint.split("step")[-1].split(".pth")[0]) if self.args.step_model_checkpoint else None

        if self.args.step_model_checkpoint == None:
            self.train_step0()
            step_checkpoint = 0
        elif step_checkpoint == 0:
            self.initialize_i_step_model(0, test=True)
            self.validate_target(0)
        
        if self.args.incremental_setup:
            if step_checkpoint > 0:
                for step in range(step_checkpoint):
                    self.next_step()
                self.initialize_i_step_model(step+1, test=True)
                self.validate_target(step+1)

            for step in range(step_checkpoint, len(self.target_train.incremental_ids_mapping)):
                self.next_step()
                self.initialize_i_step_model(step+1, test=True)
                self.validate_target(step+1)
                       
    def validate_target(self, incremental_step):
        self.model.eval()
        metrics = Metrics(self.target_val.class_names[self.target_val.step])
        with torch.no_grad():
            for x,y in tqdm(self.tloader_val, desc="Testing on Target Dataset at step {}".format(incremental_step)):
                x, y = x.to('cuda', dtype=torch.float32), y.to('cuda', dtype=torch.long)
                out = self.model(x)[0]
                pred = out.argmax(dim=1)
                metrics.add_sample(pred, y)
        self.writer.add_scalar('step%d_test_target_mIoU'%incremental_step, metrics.percent_mIoU())
        self.log_images('step%d_test_target'%incremental_step,
                        self.target_val.to_rgb(x[0].cpu()),
                        self.target_val.color_label(y[0].cpu()),
                        self.target_val.color_label(pred[0].cpu())
                        )

        self.logger.info(metrics)

    def log_images(self, prefix, rgb, gt_col, pred_col):
        self.writer.add_image(prefix+"_rgb", rgb, dataformats='HWC')
        self.writer.add_image(prefix+"_gt", gt_col, dataformats='HWC')
        self.writer.add_image(prefix+"_pred", pred_col, dataformats='HWC')

if __name__ == "__main__":
    t = Tester()
    t.test()
