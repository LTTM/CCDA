import numpy as np
import os
import torch
from tqdm import tqdm

from utils.train_initializer import Initializer
from utils.metrics import Metrics
import torch.nn.functional as F
import json

class Trainer(Initializer):
    def train(self):

        if self.args.step_model_checkpoint and ("best" in self.args.step_model_checkpoint):
            step_checkpoint = int(self.args.step_model_checkpoint.split("step")[-1].split("_best")[0])
        else:
            # extract the step from the checkpoint model provided       
            step_checkpoint = int(self.args.step_model_checkpoint.split("step")[-1].split(".pth")[0]) if self.args.step_model_checkpoint else None

        if self.args.step_model_checkpoint == None:
            self.train_step0()
            step_checkpoint = 0

        elif step_checkpoint == 0:
            self.initialize_i_step_model()

        if self.args.incremental_setup:
            if step_checkpoint > 0:
                for step in range(step_checkpoint):
                    self.next_step()
                self.initialize_i_step_model()

            for step in range(step_checkpoint, len(self.source_train.incremental_ids_mapping)):
                self.next_step()
                self.train_incremental_step()

    def train_step0(self):
        smetrics = Metrics(self.source_train.class_names[self.source_train.step])
        if self.uda:
            tmetrics = Metrics(self.target_train.class_names[self.target_train.step])
            tdata = iter(self.tloader_train)
        iters = 0
        with tqdm(total=self.args.iterations, desc="Training Step 0, loss: 0.00, Progress") as pbar:
            while iters < self.args.iterations :
                sdata = iter(self.sloader_train)
                for x, y in sdata:
                    torch.cuda.empty_cache()
                    # Supervised Training on source
                    x, y = x.to('cuda', dtype=torch.float32), y.to('cuda', dtype=torch.long)
                    self.writer.add_scalar('step0_lr', self.lr_scheduler(iters), iters) # logs and updates the learning rate
                    self.optimizer.zero_grad()
                    out = self.model(x)[0]
                    loss = self.ce_loss(out, y)
                    loss.backward()
                    pbar.set_description("Training Step 0, loss: %.2f, Progress"%loss.item())
                    pred = out.detach().argmax(dim=1)
                    smetrics.add_sample(pred, y)
                    self.writer.add_scalar('step0_ce_loss', loss.item(), iters)
                    self.writer.add_scalar('step0_train_source_mIoU', smetrics.percent_mIoU(), iters)

                    # Unsupervised training on target
                    if self.use_msiw_loss and iters > self.args.uda_skip_iterations:
                        try:
                            x, y = next(tdata)
                        except StopIteration:
                            tdata = iter(self.tloader_train) # regenerate dataloader
                            x, y = next(tdata)

                        x, y = x.to('cuda', dtype=torch.float32), y.to('cuda', dtype=torch.long)
                        out = self.model(x)[0]
                        if self.use_msiw_loss:
                            loss = self.args.msiw_lambda*self.msiw_loss(out)
                            loss.backward()                           
                            pred = out.detach().argmax(dim=1)
                            tmetrics.add_sample(pred, y)
                            self.writer.add_scalar('step0_msiw_loss', loss.item()/self.args.msiw_lambda, iters)
                            self.writer.add_scalar('step0_train_target_mIoU', tmetrics.percent_mIoU(), iters)

                    self.optimizer.step()
                    iters += 1
                    pbar.update()

                    if self.args.validate_every_steps>0 and iters>0 and iters%self.args.validate_every_steps==0:
                        self.validate_source(iters, 0)
                        if self.uda:
                            self.validate_target(iters, 0)

                    if iters >= self.args.iterations:
                        break

        self.save_model()
        if self.args.continue_from_best_model:
            print(f"Best model found at iteration {np.argmax(self.mIoU_list_per_step)*self.args.validate_every_steps+self.args.validate_every_steps}")
            self.validate_best_model(iters, self.source_train.step)

    def train_incremental_step(self):
        smetrics = Metrics(self.source_train.class_names[self.source_train.step])
        if self.uda:
            tdata = iter(self.tloader_train)
            if self.use_msiw_loss:
                tmetrics = Metrics(self.target_train.class_names[self.target_train.step])

        iters = 0
        with tqdm(total=self.args.incremental_iterations, desc="Training Step %d, loss: 0.00, Progress"%self.source_train.step) as pbar:
            while iters < self.args.incremental_iterations :
                sdata = iter(self.sloader_train)
                for x, y in sdata:
                    # Supervised Training on source
                    x, y = x.to('cuda', dtype=torch.float32), y.to('cuda', dtype=torch.long)
                    self.writer.add_scalar('step%d_lr'%self.source_train.step, self.lr_scheduler(iters, incremental=True), iters) # logs and updates the learning rate
                    self.optimizer.zero_grad()
                    out, _ = self.model(x)
                    loss = self.ce_loss(out, y)
                    loss.backward(retain_graph=(not self.use_msiw_loss and self.args.kd_lambda>0))
                    pbar.set_description("Training Step %d, ce_loss: %.2f, Progress"%(self.source_train.step, loss.item()))
                    pred = out.detach().argmax(dim=1)
                    smetrics.add_sample(pred, y)
                    self.writer.add_scalar('step%d_ce_loss'%self.source_train.step, loss.item(), iters)
                    self.writer.add_scalar('step%d_train_source_mIoU'%self.source_train.step, smetrics.percent_mIoU(), iters)

                    if  not self.use_msiw_loss and self.args.kd_lambda>0:
                        with torch.no_grad():
                            old_out, _ = self.old_model(x)

                        if (self.args.kd_lambda_c2f == self.args.kd_lambda) or self.args.kd_type=="mib":
                            kd_loss = self.args.kd_lambda*self.kd_loss(out, old_out)
                            kd_loss.backward()
                            self.writer.add_scalar('step%d_kd_loss'%self.source_train.step, kd_loss.item()/self.args.kd_lambda, iters)
                        else:
                            kd_loss_f, kf_loss_c = self.kd_loss(out, old_out)
                            kd_loss_f = kd_loss_f*self.args.kd_lambda
                            kd_loss_f.backward(retain_graph=self.args.kd_lambda_c2f>0)
                            self.writer.add_scalar('step%d_kd_loss_finefine'%self.source_train.step, kd_loss_f.item()/self.args.kd_lambda, iters)
                            if self.args.kd_lambda_c2f>0:
                                kf_loss_c = kf_loss_c*self.args.kd_lambda_c2f
                                kf_loss_c.backward()
                                self.writer.add_scalar('step%d_kd_loss_coarsefine'%self.source_train.step, kf_loss_c.item()/self.args.kd_lambda_c2f, iters)
                            else:
                                del kf_loss_c
                                torch.cuda.empty_cache() 

                    # Unsupervised training on target
                    if self.use_msiw_loss:
                        try:
                            x_t, y_t = next(tdata)
                        except StopIteration:
                            tdata = iter(self.tloader_train) # regenerate dataloader
                            x_t, y_t = next(tdata)

                        x, y = x_t.to('cuda', dtype=torch.float32), y_t.to('cuda', dtype=torch.long)
                        torch.cuda.empty_cache()
                        out, _ = self.model(x)
                        loss = self.args.msiw_lambda*self.msiw_loss(out)
                        loss.backward(retain_graph=self.args.kd_lambda>0)
                        pred = out.detach().argmax(dim=1)
                        tmetrics.add_sample(pred, y)
                        self.writer.add_scalar('step%d_msiw_loss'%self.source_train.step, loss.item()/self.args.msiw_lambda, iters)
                        self.writer.add_scalar('step%d_train_target_mIoU'%self.source_train.step, tmetrics.percent_mIoU(), iters)

                        if self.args.kd_lambda>0:
                            with torch.no_grad():
                                old_out, _ = self.old_model(x)
                            if (self.args.kd_lambda_c2f == self.args.kd_lambda) or self.args.kd_type=="mib":
                                kd_loss = self.args.kd_lambda*self.kd_loss(out, old_out)
                                kd_loss.backward()
                                self.writer.add_scalar('step%d_kd_loss'%self.source_train.step, kd_loss.item()/self.args.kd_lambda, iters)
                            else:
                                kd_loss_f, kf_loss_c = self.kd_loss(out, old_out)
                                kd_loss_f = kd_loss_f*self.args.kd_lambda
                                kd_loss_f.backward(retain_graph=self.args.kd_lambda_c2f>0)
                                self.writer.add_scalar('step%d_kd_loss_finefine'%self.source_train.step, kd_loss_f.item()/self.args.kd_lambda, iters)

                                if self.args.kd_lambda_c2f>0:
                                    kf_loss_c = kf_loss_c*self.args.kd_lambda_c2f
                                    kf_loss_c.backward()
                                    self.writer.add_scalar('step%d_kd_loss_coarsefine'%self.source_train.step, kf_loss_c.item()/self.args.kd_lambda_c2f, iters)

                    self.optimizer.step()
                    iters += 1
                    pbar.update()

                    if self.args.validate_every_steps>0 and iters>0 and iters%self.args.validate_every_steps==0:
                        self.validate_source(iters, self.source_train.step)
                        if self.uda:
                            self.validate_target(iters, self.source_train.step)

                    if iters >= self.args.incremental_iterations:
                        break

        self.save_model()
        if self.args.continue_from_best_model:
            print(f"Best model found at iteration {np.argmax(self.mIoU_list_per_step)*self.args.validate_every_steps+self.args.validate_every_steps}")
            self.validate_best_model(iters, self.source_train.step)

    def validate_best_model(self, cur_iter, incremental_step):
        if self.args.continue_from_best_model:
            best_model_path = os.path.join(self.args.logdir,
                                '%s_%s_step%d_best.pth'%(self.model.backbone_type,
                                self.model.classifier_type,
                                self.source_train.step))

            print("Best model loaded from ", best_model_path)
            checkpoint = torch.load(best_model_path)      
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.validate_source(cur_iter+1, self.source_train.step)
            if self.uda:
                self.validate_target(cur_iter+1, self.source_train.step)


    def validate_source(self, cur_iter, incremental_step):
        self.model.eval()
        metrics = Metrics(self.source_train.class_names[self.source_train.step])
        with torch.no_grad():
            for x,y in tqdm(self.sloader_val, desc="Validating on Source Dataset at Iteration %d, Progress"%cur_iter):
                x, y = x.to('cuda', dtype=torch.float32), y.to('cuda', dtype=torch.long)
                out = self.model(x)[0]
                pred = out.argmax(dim=1)
                metrics.add_sample(pred, y)
        self.writer.add_scalar('step%d_val_source_mIoU'%incremental_step, metrics.percent_mIoU(), cur_iter)
        self.log_images('step%d_val_source'%incremental_step,
                        self.source_train.to_rgb(x[0].cpu()),
                        self.source_train.color_label(y[0].cpu()),
                        self.source_train.color_label(pred[0].cpu()),
                        cur_iter)

        if (not self.args.continue_from_best_model and (cur_iter == self.args.incremental_iterations)) or (self.args.continue_from_best_model and cur_iter==self.args.incremental_iterations+1):
            self.logger.info(metrics)

        self.model.train()

    def validate_target(self, cur_iter, incremental_step):
        torch.cuda.empty_cache()
        self.model.eval()
        metrics = Metrics(self.target_train.class_names[self.target_train.step])
        with torch.no_grad():
            for x,y in tqdm(self.tloader_val, desc="Validating on Target Dataset at Iteration %d, Progress"%cur_iter):
                x, y = x.to('cuda', dtype=torch.float32), y.to('cuda', dtype=torch.long)
                out = self.model(x)[0]
                pred = out.argmax(dim=1)
                metrics.add_sample(pred, y)
        self.writer.add_scalar('step%d_val_target_mIoU'%incremental_step, metrics.percent_mIoU(), cur_iter)

        # update the best model only during the training step (at cur_iter+1 we evaluate the best model)
        max_iterations = self.args.iterations if incremental_step == 0 else self.args.incremental_iterations
        if self.args.continue_from_best_model and cur_iter <= max_iterations:
            # if the last mIoU is better than the previous ones, save the model
            self.mIoU_list_per_step.append(metrics.percent_mIoU().cpu().numpy())
            if np.max(self.mIoU_list_per_step) ==  self.mIoU_list_per_step[-1]:
                print("Updating best model, found at iteration, ", cur_iter)
                torch.save(self.model.state_dict(),
                        os.path.join(self.args.logdir,
                                        '%s_%s_step%d_best.pth'%(self.model.backbone_type,
                                        self.model.classifier_type,
                                        self.source_train.step)
                            )
                        )
        self.log_images('step%d_val_target'%incremental_step,
                        self.target_train.to_rgb(x[0].cpu()),
                        self.target_train.color_label(y[0].cpu()),
                        self.target_train.color_label(pred[0].cpu()),
                        cur_iter)

        if (not self.args.continue_from_best_model and (cur_iter == max_iterations)) or (self.args.continue_from_best_model and (cur_iter==max_iterations+1)):
            self.logger.info(metrics)
        
        self.model.train()

    def save_model(self):
        state_dict = self.model.state_dict()
        torch.save(state_dict,     
                   os.path.join(self.args.logdir,
                                '%s_%s_step%d.pth'%(self.model.backbone_type,
                                self.model.classifier_type,
                                self.source_train.step)
                    )
                )

    def log_images(self, prefix, rgb, gt_col, pred_col, iteration):
        self.writer.add_image(prefix+"_rgb", rgb, iteration, dataformats='HWC')
        self.writer.add_image(prefix+"_gt", gt_col, iteration, dataformats='HWC')
        self.writer.add_image(prefix+"_pred", pred_col, iteration, dataformats='HWC')

if __name__ == "__main__":
    t = Trainer()
    t.train()
