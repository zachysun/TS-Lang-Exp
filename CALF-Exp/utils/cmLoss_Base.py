import torch
import torch.nn.functional as F
import torch.nn as nn
from .similar_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}


class cmLoss_base(nn.Module):
    def __init__(self, task_loss, task_name):
        super(cmLoss_base, self).__init__()
        self.task_loss = loss_dict[task_loss]
        self.task_name = task_name

    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        outputs_time, intermidiate_feat_time = (
            outputs["outputs_time"],
            outputs["intermidiate_time"],
        )

        batch_y = batch_y.to(outputs_time.device)

        # supervised task loss
        if self.task_name == "long_term_forecast":
            task_loss = self.task_loss(outputs_time, batch_y)
        elif self.task_name == "short_term_forecast":
            task_loss = self.task_loss(in_sample, freq_map, outputs_time, batch_y, batch_y_mark)
        elif self.task_name == "classification":
            task_loss = self.task_loss(outputs_time, batch_y)
        elif self.task_name == "imputation":
            task_loss = self.task_loss(outputs_time, batch_y)
        elif self.task_name == "anomaly_detection":
            task_loss = self.task_loss(outputs_time, batch_y)

        total_loss = task_loss
        return total_loss
