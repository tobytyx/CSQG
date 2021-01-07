# -*- coding: utf-8 -*-
from torch.optim.lr_scheduler import LambdaLR


class MySchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to min_lr over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(MySchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.01, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class MyMSSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate is milestone.
    """
    def __init__(self, optimizer, warmup_steps, milestones, gamma=0.1):
        self.warmup_steps = warmup_steps
        self.milestones = list(milestones)
        self.gamma = gamma
        super(MyMSSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.gamma ** bisect_right(self.milestones, step)


def bisect_right(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid+1
    return lo
