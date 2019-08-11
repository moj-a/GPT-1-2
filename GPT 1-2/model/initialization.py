import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math



            
            
            
def GPT1_weight_init(m):
    """
    Initialization weights of linear layers with normal N(0, 0.02) (based on GPT1 initialization)
    :param m: nn.module
    :To initialize all the weights of our model we can use: model.apply(GPT1_weight_init)
    """

    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0, std=0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)   


            
def xavier_normal_init(m):
    """
    Initialization weight of linear layers with xavier_normal N(0, std): http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    :param m: nn.module
    We can initialize biases to zero, since the gradients with respect to bias depend only on the linear activation of that layer, and not on the gradients of the deeper           layers. Thus there is no diminishing or explosion of gradients for the bias terms. 
    """

    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
            #init.normal_(m.bias.data)
            
            
            
def kaiming_normal_init(m):
    """
    Initialization weights and biases of only linear layers with kaiming_normal N(0, std) https://arxiv.org/pdf/1502.01852.pdf
    weights are initialized using Kaiming Uniform method. 
    Bias are initialized using LeCunn init (uniform(-std, std), where standard deviation std is 1/sqrt(fan_in))  http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    :param m: nn.module
    """

    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)          
            
            
            
#usefull reading to understadn the difference between Xavier and Kaiming: https://pouannes.github.io/blog/initialization/