#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   callbacks.py
@Time    :   2022/03/16 23:30:24
@Author  :   weiguang 
'''

import torch
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import History

EarlyStopping = EarlyStopping
History = History

class ModelCheckpoint(ModelCheckpoint):
    """
    每一轮epoch保存模型
    filepath是按照epoch轮数及验证集上loss值
    例如: weights.{epoch:02d}-{val_loss:.2f}.hdf5

    Args:
        filepath: 保存模型的路径
        monitor: 监视器类型, 如auc与acc
        save_best_only: 如果为真,则保留最新的效果最好的model
        mode: [auto, min, max]
        save_weights_only: 只保存模型的weights信息
        period: checkpoints的间隔(epochs的数量)
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # 最后保存的epochs数量
        self.epochs_since_last_save += 1
        # 如果当前最后保存的epochs数量 > 
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch+1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    print('Can save best model only with %s available, skipping.' % self.monitor)
                else:
                    # 监控当前与best
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        # 如果是只保存weights
                        if self.save_weights_only:
                            torch.save(self.model.state_dict(), filepath)
                        else:
                            torch.save(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save(self.model, filepath)

