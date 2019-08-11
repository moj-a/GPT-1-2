from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import math


increasing_size=2000

class GPT1CosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, increasing_size=increasing_size):
        """
        Learning rate schedule based on GPT1 paper: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

        Learning rate is increasing linearly from zero over the first 2000 (=increasing_size) updates and annealed to 0 using a cosine schedule. 
        The cosine schedule part is based on CosineAnnealingLR from Pytorch and the paper : https://arxiv.org/pdf/1608.03983.pdf

        :param optimizer: Pytorch optimizer object.
        :param T_max (int): Maximum numnber of iteration in cosine schedule part (half of period). 
        :param eta_min (float): Minimum learning rate. Default: 0.
        :param last_epoch (int) – The index of last step. Default: -1.
        :param increasing_size: The number of steps that learming rate is increasing.
        
        """
        
        self.increasing_size = increasing_size
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.increasing_size:
            learN = [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch-self.increasing_size) / self.T_max)) / 2
                for base_lr in self.base_lrs]
        
        else:
            learN = [(base_lr/ self.increasing_size) * (self.last_epoch) for base_lr in self.base_lrs] 
        
        
        return learN
    
    
    
    
    
    
    
    
    
#The equvalent wayto do it using Ramped method:    
    
class RampedCosineAnnealingLR(CosineAnnealingLR):
    
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, increasing_size=increasing_size):
        
        super().__init__(optimizer, T_max, eta_min=eta_min, last_epoch=last_epoch)
        self.increasing_size = increasing_size
        
    def get_lr(self):
        if self.last_epoch > self.increasing_size:
            learN = [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch-self.increasing_size) / self.T_max)) / 2
                for base_lr in self.base_lrs]
        
        else:
            learN = [(base_lr/ self.increasing_size) * (self.last_epoch) for base_lr in self.base_lrs] 
        
        
        return learN