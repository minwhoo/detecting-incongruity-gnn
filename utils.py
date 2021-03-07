# -*- coding: utf-8 -*-

from pathlib import Path
import datetime
import torch

CHECKPOINT_SAVE_DIR = Path('/tmp')


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.min_loss = float("inf")
        self.checkpoint_save_path = None

    def record_loss(self, loss, model):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
            checkpoint_file_name = f"checkpoint_{datetime.datetime.now()}.pt".replace(' ', '_')
            if self.checkpoint_save_path is not None:
                self.checkpoint_save_path.unlink()
            self.checkpoint_save_path = CHECKPOINT_SAVE_DIR / checkpoint_file_name
            torch.save(model.state_dict(), self.checkpoint_save_path)
        else:
            self.counter += 1

    def should_stop(self):
        return self.counter >= self.patience
