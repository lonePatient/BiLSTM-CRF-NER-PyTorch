#encoding:utf-8
import os
import time
import numpy as np
import torch
from ..callback.progressbar import ProgressBar
from ..utils.utils import AverageMeter
from .train_utils import restore_checkpoint,model_device
from .metrics import Entity_Score
from .train_utils import batchify_with_label

# 训练包装器
class Trainer(object):
    def __init__(self,model,
                 train_data,
                 val_data,
                 optimizer,
                 epochs,
                 logger,
                 evaluate,
                 avg_batch_loss   = False,
                 label_to_id      = None,
                 n_gpu            = None,
                 lr_scheduler     = None,
                 resume           = None,
                 model_checkpoint = None,
                 training_monitor = None,
                 early_stopping   = None,
                 writer           = None,
                 verbose = 1):
        self.model            = model              # 模型
        self.train_data       = train_data         # 训练数据
        self.val_data         = val_data           # 验证数据
        self.epochs           = epochs             # epochs次数
        self.optimizer        = optimizer          # 优化器
        self.logger           = logger             # 日志记录器
        self.verbose          = verbose            # 是否打印
        self.writer           = writer             # tensorboardX写入器
        self.training_monitor = training_monitor   # 监控训练过程指标变化
        self.early_stopping   = early_stopping     # early_stopping
        self.resume           = resume             # 是否重载模型
        self.model_checkpoint = model_checkpoint   # 模型保存
        self.lr_scheduler     = lr_scheduler       # 学习率变化机制，这里需要根据不同机制修改不同代码
        self.evaluate         = evaluate           # 评估指标
        self.n_gpu            = n_gpu              # gpu个数，列表形式
        self.avg_batch_loss   = avg_batch_loss     # 建议大的batch_size使用loss avg
        self.id_to_label      = {value:key for key,value in label_to_id.items()}
        self._reset()

    def _reset(self):

        self.train_entity_score = Entity_Score(id_to_label=self.id_to_label)
        self.val_entity_score   = Entity_Score(id_to_label=self.id_to_label)

        self.batch_num         = len(self.train_data)
        self.progressbar       = ProgressBar(n_batch = self.batch_num,eval_name='acc',loss_name='loss')
        self.model,self.device = model_device(n_gpu=self.n_gpu,model = self.model,logger = self.logger)
        self.start_epoch = 1

        # 重载模型，进行训练
        if self.resume:
            arch = self.model_checkpoint.arch
            resume_path = os.path.join(self.model_checkpoint.checkpoint_dir.format(arch = arch),
                                       self.model_checkpoint.best_model_name.format(arch = arch))
            self.logger.info("\nLoading checkpoint: {} ...".format(resume_path))
            resume_list = restore_checkpoint(resume_path = resume_path,model = self.model,optimizer = self.optimizer)
            self.model     = resume_list[0]
            self.optimizer = resume_list[1]
            best           = resume_list[2]
            self.start_epoch = resume_list[3]

            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info("\nCheckpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # for p in model_parameters:
        #     print(p.size())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # 总的模型参数量
        self.logger.info('trainable parameters: {:4}M'.format(params / 1000 / 1000))
        # 模型结构
        self.logger.info(self.model)

    # 保存模型信息
    def _save_info(self,epoch,val_loss):
        state = {
            'epoch': epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': round(val_loss,4)
        }
        return state

    # val数据集预测
    def _valid_epoch(self):
        self.model.eval()
        val_losses = AverageMeter()
        val_acc    = AverageMeter()
        val_f1     = AverageMeter()
        self.val_entity_score._reset()
        with torch.no_grad():
            for batch_idx, (inputs,target,length) in enumerate(self.val_data):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                length = length.to(self.device)
                batch_size = inputs.size(0)

                outputs = self.model(inputs, length)
                mask,target = batchify_with_label(inputs = inputs,target = target,outputs = outputs)
                loss = self.model.crf.neg_log_likelihood_loss(outputs, mask,target)
                if self.avg_batch_loss:
                    loss /=  batch_size
                _,predicts = self.model.crf(outputs, mask)
                acc,f1 = self.evaluate(predicts,target=target)

                val_losses.update(loss.item(),batch_size)
                val_acc.update(acc.item(),batch_size)
                val_f1.update(f1.item(),batch_size)
                if self.device != 'cpu':
                    predicts = predicts.cpu().numpy()
                    target = target.cpu().numpy()
                self.val_entity_score.update(pred_paths=predicts, label_paths=target)

        return {
            'val_loss': val_losses.avg,
            'val_acc': val_acc.avg,
            'val_f1': val_f1.avg
        }

    # epoch训练
    def _train_epoch(self):
        self.model.train()
        train_loss = AverageMeter()
        train_acc  = AverageMeter()
        train_f1   = AverageMeter()
        self.train_entity_score._reset()
        for batch_idx, (inputs,target,length) in enumerate(self.train_data):
            start    = time.time()
            inputs   = inputs.to(self.device)
            target   = target.to(self.device)
            length   = length.to(self.device)
            batch_size = inputs.size(0)

            outputs = self.model(inputs,length)
            mask, target = batchify_with_label(inputs=inputs, target=target, outputs=outputs)
            loss    = self.model.crf.neg_log_likelihood_loss(outputs,mask,target)
            if self.avg_batch_loss:
                loss  /= batch_size

            _,predicts = self.model.crf(outputs,mask)
            acc,f1 = self.evaluate(predicts,target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(),batch_size)
            train_acc.update(acc.item(),batch_size)
            train_f1.update(f1.item(),batch_size)

            if self.device != 'cpu':
                predicts = predicts.cpu().numpy()
                target   = target.cpu().numpy()
            self.train_entity_score.update(pred_paths=predicts,label_paths=target)

            if self.verbose >= 1:
                self.progressbar.step(batch_idx=batch_idx,
                                      loss     = loss.item(),
                                      acc      = acc.item(),
                                      f1       = f1.item(),
                                      use_time = time.time() - start)
        print("\ntraining result:")
        train_log = {
            'loss': train_loss.avg,
            'acc': train_acc.avg,
            'f1': train_f1.avg
        }
        return train_log

    def train(self):
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):

            print("----------------- training start -----------------------")
            print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch+self.epochs -1))

            train_log = self._train_epoch()
            val_log = self._valid_epoch()

            logs = dict(train_log,**val_log)
            self.logger.info('\nEpoch: %d - loss: %.4f acc: %.4f - f1: %.4f val_loss: %.4f - val_acc: %.4f - val_f1: %.4f'%(
                            epoch,logs['loss'],logs['acc'],logs['f1'],logs['val_loss'],logs['val_acc'],logs['val_f1'])
                             )
            print("----------- Train entity score:")
            self.train_entity_score.result()
            print("----------- valid entity score:")
            self.val_entity_score.result()

            if self.lr_scheduler:
                self.lr_scheduler.step(logs['loss'],epoch)

            if self.training_monitor:
                self.training_monitor.step(logs)

            if self.model_checkpoint:
                state = self._save_info(epoch,val_loss = logs['val_loss'])
                self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor],state = state)

            if self.writer:
                self.writer.set_step(epoch,'train')
                self.writer.add_scalar('loss', logs['loss'])
                self.writer.add_scalar('acc', logs['acc'])
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', logs['val_loss'])
                self.writer.add_scalar('val_acc', logs['val_acc'])

            if self.early_stopping:
                self.early_stopping.step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break

