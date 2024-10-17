import sys
import json
import os
import time
from datetime import datetime
import math
import sys
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorama
from colorama import Back, Fore, Style
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path

from utils.core_utils import (
    get_collator,
    get_dataset,
    load_model,
    set_seed,
    set_worker_seed,
    get_optimizer,
    get_scheduler,
    BinaryClassificationMetric,
    TernaryClassificationMetric,
    EarlyStopping
)


log_path = Path(f'log/{datetime.now().strftime("%m%d-%H%M%S")}')


class Trainer():
    def __init__(self,
                 cfg: DictConfig):
        self.cfg = cfg
        
        self.device = 'cuda'
        self.task = cfg.task
        if cfg.task == 'binary':
            self.evaluator = BinaryClassificationMetric(self.device)
        elif cfg.task == 'ternary':
            self.evaluator = TernaryClassificationMetric(self.device)
        else:
            raise ValueError('task not supported')
        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.save_path = log_path
        
        if cfg.type == 'default':
            self.dataset_range = ['default']
        else:
            raise ValueError('experiment type not supported')
        
        self.collator = get_collator(cfg.model, cfg.dataset, **cfg.data)
    
    def _reset(self, cfg, fold, type):
        train_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='train', **cfg.data)
        if hasattr(cfg, 'general') and cfg.general:
            logger.info(f"Using {cfg.general.dataset} as test dataset!")
            test_dataset = get_dataset(cfg.model, cfg.general.dataset, fold=fold, split='test', **cfg.data)
        else:
            test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='test', **cfg.data)
        if cfg.task == 'binary':
            valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='valid', **cfg.data)
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=True, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
        if cfg.task == 'binary':
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))

        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        self.model = load_model(cfg.model, **dict(cfg.para))
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, **dict(cfg.sche))
        self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path/'best_model.pth')
        
    def run(self):
        acc_list, f1_list, prec_list, rec_list = [], [], [], []
        a_f1_list, a_prec_list, a_rec_list = [], [], []
        b_f1_list, b_prec_list, b_rec_list = [], [], []
        c_f1_list, c_prec_list, c_rec_list = [], [], []
        for fold in self.dataset_range:
            self._reset(self.cfg, fold, self.type)
            logger.info(f'Current fold: {fold}')
            for epoch in range(self.num_epoch):
                logger.info(f'Current Epoch: {epoch}')
                self._train(epoch=epoch)
                if self.task == 'binary':
                    self._valid(split='valid', epoch=epoch, use_earlystop=True)
                    if self.earlystopping.early_stop:
                        logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                        break
                    self._valid(split='test', epoch=epoch)
                elif self.task == 'ternary':
                    self._valid(split='test', epoch=epoch, use_earlystop=True)
                    if self.earlystopping.early_stop:
                        logger.info(f"{Fore.RED}Early stopping at epoch {epoch}")
                        break
            logger.info(f'{Fore.RED}Best of Acc in fold {fold}:')
            self.model.load_state_dict(torch.load(self.save_path/'best_model.pth', weights_only=False))
            best_metrics = self._valid(split='test', epoch=epoch, final=True)
            acc_list.append(best_metrics['acc'])
            f1_list.append(best_metrics['macro_f1'])
            prec_list.append(best_metrics['macro_prec'])
            rec_list.append(best_metrics['macro_rec'])
            a_f1_list.append(best_metrics['a_f1'])
            a_prec_list.append(best_metrics['a_prec'])
            a_rec_list.append(best_metrics['a_rec'])
            b_f1_list.append(best_metrics['b_f1'])
            b_prec_list.append(best_metrics['b_prec'])
            b_rec_list.append(best_metrics['b_rec'])
            if self.task == 'ternary':
                c_f1_list.append(best_metrics['c_f1'])
                c_prec_list.append(best_metrics['c_prec'])
                c_rec_list.append(best_metrics['c_rec'])
            
        logger.info(f'Best of Acc in all fold: {np.mean(acc_list)}, Best F1: {np.mean(f1_list)}, Best Precision: {np.mean(prec_list)}, Best Recall: {np.mean(rec_list)}')
        logger.info(f'Best of A F1 in all fold: {np.mean(a_f1_list)}, Best A Precision: {np.mean(a_prec_list)}, Best A Recall: {np.mean(a_rec_list)}')
        logger.info(f'Best of B F1 in all fold: {np.mean(b_f1_list)}, Best B Precision: {np.mean(b_prec_list)}, Best B Recall: {np.mean(b_rec_list)}')
        if self.task == 'ternary':
            logger.info(f'Best of C F1 in all fold: {np.mean(c_f1_list)}, Best C Precision: {np.mean(c_prec_list)}, Best C Recall: {np.mean(c_rec_list)}')
            
    def _train(self, epoch: int):
        loss_list =  []
        loss_pre_list = []
        self.model.train()
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        for batch in pbar:
            _ = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')
            
            output = self.model(**inputs)
            pred = output['pred'] if isinstance(output, dict) else output
            
            match self.model.name:
                case 'MoRE':
                    loss, loss_pred = self.model.calculate_loss(**output, label=labels, epoch=epoch)
                case _:
                    loss = F.cross_entropy(pred, labels)
                    loss_pred = loss

            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
            loss_pre_list.append(loss_pred.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        metrics = self.evaluator.compute()
        # print
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}")
        
        logger.info(f'{Fore.BLUE}Train: Acc: {metrics["acc"]:.5f}, Macro F1: {metrics["macro_f1"]:.5f}, Macro Prec: {metrics["macro_prec"]:.5f}, Macro Rec: {metrics["macro_rec"]:.5f}')
        logger.info(f'{Fore.BLUE}Train: A F1: {metrics["a_f1"]:.5f}, A Prec: {metrics["a_prec"]:.5f}, A Rec: {metrics["a_rec"]:.5f}')
        logger.info(f'{Fore.BLUE}Train: B F1: {metrics["b_f1"]:.5f}, B Prec: {metrics["b_prec"]:.5f}, B Rec: {metrics["b_rec"]:.5f}')
        if self.task == 'ternary':
            logger.info(f'{Fore.BLUE}Train: C F1: {metrics["c_f1"]:.5f}, C Prec: {metrics["c_prec"]:.5f}, C Rec: {metrics["c_rec"]:.5f}')
    
    def _valid(self, split: str, epoch: int, use_earlystop=False, final=False):
        loss_list = []
        self.model.eval()
        if split == 'valid' and final:
            raise ValueError('print_wrong only support test split')
        if split == 'valid':
            dataloader = self.valid_dataloader
            split_name = 'Valid'
            fcolor = Fore.YELLOW
        elif split == 'test':
            dataloader = self.test_dataloader
            split_name = 'Test'
            fcolor = Fore.RED
        else:
            raise ValueError('split not supported')
        for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')
        
            with torch.no_grad():
                output = self.model(**inputs)
                pred = output['pred'] if isinstance(output, dict) else output
                loss = F.cross_entropy(pred, labels)
            
            _, preds = torch.max(pred, 1)

            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
        metrics = self.evaluator.compute()
        
        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list):.5f}")
        logger.info(f"{fcolor}{split_name}: Acc: {metrics['acc']:.5f}, Macro F1: {metrics['macro_f1']:.5f}, Macro Prec: {metrics['macro_prec']:.5f}, Macro Rec: {metrics['macro_rec']:.5f}")
        logger.info(f"{fcolor}{split_name}: A F1: {metrics['a_f1']:.5f}, A Prec: {metrics['a_prec']:.5f}, A Rec: {metrics['a_rec']:.5f}")
        logger.info(f"{fcolor}{split_name}: B F1: {metrics['b_f1']:.5f}, B Prec: {metrics['b_prec']:.5f}, B Rec: {metrics['b_rec']:.5f}")
        if self.task == 'ternary':
            logger.info(f"{fcolor}{split_name}: C F1: {metrics['c_f1']:.5f}, C Prec: {metrics['c_prec']:.5f}, C Rec: {metrics['c_rec']:.5f}")
        if use_earlystop:
            if self.task == 'binary':
                self.earlystopping(metrics['acc'], self.model)
            else:
                raise ValueError('task not supported')
        return metrics

@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    logger.remove()
    logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    pd.set_option('future.no_silent_downcasting', True)
    colorama.init()
    set_seed(cfg.seed)
    
    trainer = Trainer(cfg)
    trainer.run()

if  __name__ == '__main__':
    main()