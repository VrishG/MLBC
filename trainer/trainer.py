import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', 'accuracy', 'precision', 'recall', 'f1', writer=self.writer)
        self.valid_metrics = MetricTracker('val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('train_loss', loss.item(), (epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            # Calculate metrics
            preds = torch.argmax(output, dim=1)
            accuracy = accuracy_score(target.cpu(), preds.cpu())
            precision = precision_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)
            recall = recall_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)

            self.train_metrics.update('accuracy', accuracy)
            self.train_metrics.update('precision', precision)
            self.train_metrics.update('recall', recall)
            self.train_metrics.update('f1', f1)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Accuracy: {:.6f} Precision: {:.6f} Recall: {:.6f} F1: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    accuracy,
                    precision,
                    recall,
                    f1
                ))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)  # Update with validation metrics

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.add_scalar('val_loss', loss.item(), (epoch - 1) * len(self.valid_data_loader) + batch_idx)
                self.valid_metrics.update('val_loss', loss.item())

                # Calculate metrics
                preds = torch.argmax(output, dim=1)
                accuracy = accuracy_score(target.cpu(), preds.cpu())
                precision = precision_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)
                recall = recall_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)
                f1 = f1_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)

                self.valid_metrics.update('val_accuracy', accuracy)
                self.valid_metrics.update('val_precision', precision)
                self.valid_metrics.update('val_recall', recall)
                self.valid_metrics.update('val_f1', f1)

                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
