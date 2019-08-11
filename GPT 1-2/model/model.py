import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os


# -- activation functions --

def gelu(x):
    """
    GAUSSIAN ERROR LINEAR UNITS (GELUS)
    Paper: https://arxiv.org/pdf/1606.08415.pdf

    :param x: torch.Tensor of any shape
    :return: torch.Tensor same shape as input
    """
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


# -- model components --

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, activation=F.relu, dropout_value=0.1):
        """
        Position-Wise Feed Forward layer as described by paper:  Attention is all you need

        :param d_model: dimension of the embedded text
        :param d_ff: internal dimension of the P-W Feed Forward layer
        :param activation: activation function to use
        :param dropout_value: dropout percentage applied to output of final layer
        """
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = activation
        self.drop = nn.Dropout(p=dropout_value)

    def forward(self, x):
        """
        Forward pass through Position-Wise Feed Forward layer
        :param x: torch.Tensor of shape [B, **, d_model]
        :return: torch.Tensor of shape [B, **, d_model]
        """
        x = self.w2(self.act(self.w1(x)))
        x = self.drop(x)
        return x
    
    
#only for debugging:    
def log_tensor_shape(name, tensor):
    tensor_shape = tensor.size()
    
    with open(name+".txt", "a+") as text_file:
        text_file.write(f"Shape: {tensor_shape} \n")
        
        
def log_tensor(name, tensor):
    
    with open(name+".txt", "a+") as text_file:
        text_file.write(f"Shape: {tensor} \n")       
#----------------------------------------        
        

def attention(query, key, value, mask=True):
    """
    Scaled dot-product attention from Paper: Attention is all you need 

    :param tensor query: tensor of shape [B, n, d_k], from decoder
    :param tensor key: tensor of shape [B, m, d_k], from encoder
    :param tensor value: tensor of shape [B, m, d_v], from encoder
    :param bool mask: whether to apply a mask before softmax
    :return: attention tensor of shape [B, n, d_v]
    """

    d_k = key.size(-1)
    first_product = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    

    
    
    try:
        if mask is True:
            attn_mask = np.triu(np.ones(first_product.shape), k=1)
            #TODO: Fix masking for general case CUDA vs CPU
            # get device from input (cpu or cuda)
            device = query.device



            attn_mask = torch.from_numpy(attn_mask).byte().to(device)
            first_product = first_product.masked_fill_(attn_mask, -np.inf)

        atten = F.softmax(first_product, dim=-1)
        output = torch.matmul(atten, value)
    except:
            print("attention mask:")
            #print("first_product:", first_product)
            #print("query", query)
            #print("key", key)
            print(" - size: ", attn_mask.shape)
            print(" - device: ", device)
            print(" - array: ", attn_mask)
            raise ValueError('Attention ERROR')
            # --- debugging-- 
    
    return output, atten


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=24, h=12, dropout_value=0.1, mask=True):
        """
        Multi-head_attention from Paper: Attention is all you need
        
        :param int d_model: sets the dimension of the embeddings for each token 
        :param int h: number of heads 
        :param float dropout_value: dropout percentage for final layer
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        # store variables
        self.d_v = int(d_model / h)
        self.h = h
        self.d_model = d_model

        # create projection layers
        self.w_Q = nn.Linear(d_model, d_model)
        self.w_K = nn.Linear(d_model, d_model)
        self.w_V = nn.Linear(d_model, d_model)

        # create output layer and dropout 
        self.w_O = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p=dropout_value)

        # store attention value 
        self.attention = None
        self.mask = mask

    def forward(self, Q, K, V):
        """
        Forward pass through Multi-head_attention
        :param tensor Q: tensor of shape [B, n, d_k], from decoder
        :param tensor K: tensor of shape [B, m, d_k], from encoder
        :param tensor V: tensor of shape [B, m, d_v], from encoder

        :param bool mask: whether to apply a mask before softmax
        :return: attention tensor of shape [B, n, d_model]
        """
        # get size of inputs Q, K, V size: [B, n, d_model]
        batch_size = Q.size(0)
        n = Q.size(1)

        # project Q, K, V using a linear layer [B, n, d_model] -> [B, n, d_model]
        query = self.w_Q(Q)
        key = self.w_Q(K)
        value = self.w_Q(V)

        # split the query, key, value into h separate inputs in a new dimension, then transpose
        # [B, n, d_model] -> [B, n, h, d_model/h] transpose--> [B, h, n, d_model/h]
        query = query.view(batch_size, n, self.h, self.d_v).transpose(-3, -2)
        key = key.view(batch_size, n, self.h, self.d_v).transpose(-3, -2)
        value = value.view(batch_size, n, self.h, self.d_v).transpose(-3, -2)

        # compute self-attention using the query, key, value. Output shape [B, h, n, d_model/h]
        # Note: for multi-head self-attention d_k = d_v = d_model/h
        x, self.attention = attention(query, key, value, self.mask)

        # reverse transpose then combine h and d_model/h dimension
        # [B, h, n, d_model/h] transpose-->  [B, n, h, d_model/h] --> [B, n, d_model]
        x = x.transpose(-3, -2).contiguous().view(batch_size, n, self.d_model)

        # apply dropout
        x = self.drop(x)
        # final output linear layer [B, n, d_model] -> [B, n, d_model]
        x = self.w_O(x)
        return x


class LayerNormInit(nn.Module):

    def __init__(self, normalized_shape, weight_init=1, bias_init=0, eps=1e-5, elementwise_affine=True):
        """
        Create a layer norm initialised with W and B
        :param normalized_shape: input shape from an expected input of size (See docs for nn.LayerNorm)
        :param weight_init: integer to initialize weight parameter with
        :param bias_init: integer to initialize bias parameter with
        :param eps:  a value added to the denominator for numerical stability. Default: 1e-5
        :param elementwise_affine: a boolean value that when set to True, this module has learnable per-element
        affine parameters initialized to ones (for weights) and zeros (for biases). Default: True.
        """
        super(LayerNormInit, self).__init__()

        # Create a layer norm layer
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

        # Adjust weights and bias
        self.layer_norm.weight.data.mul_(weight_init)
        self.layer_norm.bias.data.add_(bias_init)

    def forward(self, x):
        """
        Forward pass through LayerNorm
        :param x: torch.tensor of shape [B, **, "normalized_shape"]
        :return: torch.tensor of shape [B, **, "normalized_shape"]
        """
        return self.layer_norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, activation=F.relu, h=12, dropout_value=0.1, weight_init=1, bias_init=0,
                 mask=True):
        """
        Transformer block from Paper: Attention is all you need

        :param int d_model: sets the dimension of the embeddings for each token
        :param int h: number of heads
        :param float dropout_value: dropout percentage for final layer
        :param d_ff: inner-layer dimensionality in Feed-forward network
        :param F.relu: acctivation layer in Feed-forward network
        :param weight_init: integer to initialize weight parameter with in LayerNorm
        :param bias_init: integer to initialize bias parameter with in LayerNorm
        :param mask: bool value in multi_head_attention
        """

        super(TransformerBlock, self).__init__()
        self.multi = MultiHeadAttention(d_model=d_model, h=h, dropout_value=dropout_value, mask=mask)
        self.FeedFor = FeedForward(d_model=d_model, d_ff=d_ff, activation=activation, dropout_value=dropout_value)
        self.LayNorm1 = nn.LayerNorm(d_model)#, weight_init=weight_init, bias_init=bias_init)
        self.LayNorm2 = nn.LayerNorm(d_model)#, weight_init=weight_init, bias_init=bias_init)



    def forward(self, x):
        """
        Forward pass through Self-Attention Multi-Head Attention
        :param tensor X: torch.Tensor of shape [B, n, d_model], from decoder
        :return: torch.Tensor of shape [B, n, d_model]
        """

        # residual connection between x and multi-head attention [B, n, d_model]
        residual_1 = self.multi(x, x, x) + x

        # layer norm [B, n, d_model]
        norm_output = self.LayNorm1(residual_1)

        # residual connection between layerNorm and FeedForward [B, n, d_model]
        feed_output_2 = self.FeedFor(norm_output) + norm_output

        # layer norm [B, n, d_model]
        output = self.LayNorm2(feed_output_2)

        return output


# -- Embeddings --

class EmbeddingScaled(nn.Module):

    def __init__(self, nb_tokens, d_model):
        """
        Embedding layer scaled by sqrt(d_model)
        :param int nb_tokens: total number of tokens in language (which is different from n, the number of tokens in
        our text)
        :param int d_model: dimensionality of the model embeddings
        """
        super(EmbeddingScaled, self).__init__()

        self.d_model = d_model
        self.emb = nn.Embedding(nb_tokens, d_model)

    def forward(self, x):
        """
        Forward pass through embedding
        :param x: torch.Tensor of shape [B, n]
        :return: torch.Tensor of shape [B, n, d_model]
        """
        return self.emb(x) * np.sqrt(self.d_model)


def create_shared_weights_emb(nb_tokens, d_model):
    """
    Create shared weights across the encoder and the pre-softmax linear layer from Paper:
    https://arxiv.org/abs/1608.05859

    :param nb_tokens: total number of tokens
    :param d_model: dimensionality of the model embeddings
    :return: list of torch nn.Modules [encoder, pre-softmax linear] with shared weights
    """

    # investigate uniform init (defaults to normal)
    encoder = EmbeddingScaled(nb_tokens, d_model)
    pre_softmax_linear = nn.Linear(d_model, nb_tokens, bias=False)

    # link weights from encoder and decoder
    pre_softmax_linear.weight = encoder.emb.weight

    return (encoder, pre_softmax_linear)


def positional_encoding(x):
    """
    Positional Encoding from Paper: Attention is all you need
    :param tensor X: torch.Tensor of shape [B, n, d_model]
    :return: torch.Tensor of shape [B, n, d_model]
    """
    # get size from input
    d_model = x.size(-1)
    n = x.size(-2)

    # define position encoding matrix with all elements of zeros
    PE = torch.zeros(n, d_model)

    # create position from 0 to n and I from 0 to d_model
    pos = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    I = torch.arange(0, d_model / 2, dtype=torch.float)

    # power_part is the denominator of the equations in page 6 of "Attenton is All You Need"
    power_part = 10000 ** (2 * I / d_model)

    # apply sin function to all even columns
    PE[:, 0::2] = torch.sin(pos / power_part)

    # apply cos function to all odd columns
    # when d_model is odd, we should drop the last element of power_part to find a coorect size of PE
    if d_model % 2 == 0:
        PE[:, 1::2] = torch.cos(pos / power_part)
    else:
        PE[:, 1::2] = torch.cos(pos / power_part[:-1])

    return PE


class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, max_n, d_model):
        """
        Learned positional embedding
        :param max_n: maximum sequence length tokens
        :param d_model: dimensionality of the embedding
        """
        super(LearnedPositionalEmbedding, self).__init__()

        # create a weight parameter of size [max_n, d_model]
        self.max_n = max_n
        self.positional_emb = nn.Parameter(torch.ones(max_n, d_model))

        # initialise embedding (same init as nn.Embedding layer)
        torch.nn.init.normal_(self.positional_emb)

    def forward(self, x):
        """
        Forward pass of Learned Positional Embedding, adding the positional embedding to x
        :param x: torch.Tensor of size [B, **, n, d_model] note that n must be <= max_n
        :return: torch.Tensor of size [B, **, n, d_model]
        """
        n = x.size(-2)

        if n > self.max_n:
            raise ValueError('Input must be of shape [B, **, n, d_model] where n is <= max_n ({max_n})'.format(
                max_n=self.max_n))

        return x + self.positional_emb[:n]

    
    
#----Transformer-------

class Transformer(nn.Module):
    def __init__(self, max_n=512, nb_tokens=50256, d_model=768, d_ff=3072, activation=gelu, h=12, dropout_value=0.1, weight_init=0, bias_init=0.02,
        mask=True, n_block = 12):
        """
        Transformer from Paper: GPT1

        :param int d_model: sets the dimension of the embeddings for each token
        :param int h: number of heads
        :param float dropout_value: dropout percentage for final layer
        :param d_ff: inner-layer dimensionality in Feed-forward network
        :param F.relu: acctivation layer in Feed-forward network
        :param weight_init: integer to initialize weight parameter with in LayerNorm
        :param bias_init: integer to initialize bias parameter with in LayerNorm
        :param mask: bool value in multi_head_attention
        """

        super(Transformer, self).__init__()
        
        
        self.max_n = max_n
        
        self.n_block = n_block
        
        self.LearnPos = LearnedPositionalEmbedding(max_n=max_n, d_model=d_model)

        self.TranBlock = nn.ModuleList()
        for _ in range(n_block):
            self.TranBlock.append(TransformerBlock(d_model=d_model, d_ff=d_ff, activation=activation, h=h, dropout_value=dropout_value, weight_init=weight_init, bias_init=bias_init,
            mask=mask))

        self.EmbScal, self.linear = create_shared_weights_emb(nb_tokens=nb_tokens, d_model=d_model)
        
        #just for test, delet later and add self.linear instead of _
        #self.linear= nn.Linear(d_model, nb_tokens)


        

    def forward(self, x):
        """
        Forward pass through Transformer
        :param tensor X: torch.Tensor of shape [B, n]
        :return: torch.Tensor of shape [B, n, nb_tokens]
        """


        #Token embedding 
        embd_output = self.EmbScal(x) 

        #Position embedding
        x = self.LearnPos(embd_output)
    
        # Subsequent 12 Transformer block [B, n, d_model]
        for i in range(self.n_block):
            x = self.TranBlock[i](x)

        # linear layer  [B, n, d_model] -> [B, n, nb_tokens]
        linear_output = self.linear(x)
        

        # Softmax [B, n, nb_tokens]
        output = F.log_softmax(linear_output, dim=-1)

        return output


    #----inference-------
    def generate(self, inputs, max_pred): 
        """
        :param inputs: input of size [B, n]
        :param int max_ped: token lenght of the prediction 
        :param output: size of [B, n+max_ped]
        :param max_n: maximum sequence in training]
        """
        
        
        prediction=inputs
        
        
        #adjust the inputs size baed on max_n
        inputs=inputs[:,-self.max_n:]
        

        for i in range(max_pred):

            #[B, n]->[B, n, nb_tokens]       
            output = self.forward(inputs)

            #indices is in shape of [B, n]
            # sample from the output using a greedy method 
            _,indices= torch.max(output, 2)
            
            #[B, n]->[B, 1]
            indi= indices[:,-1].unsqueeze(-1)

            #[B, m]+[B, 1]->[B, m+1]
            prediction = torch.cat((prediction, indi), -1)         
            inputs=prediction[:,-self.max_n:]  

                     

        return prediction

