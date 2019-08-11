import matplotlib.pyplot as plt
import torch


def matrix_band_part(input, num_lower, num_upper):
    """
    Copy a tensor setting everything outside a central band in each innermost matrix to zero.
    Implemented based on tensorflow implementation of "tf.matrix_band_part"
    Docs: https://www.tensorflow.org/api_docs/python/tf/linalg/band_part
    :param torch.Tensor input: [B, **, M, N]
    :param int num_lower:  Number of subdiagonals to keep. If negative, keep entire lower triangle.
    :param int num_upper: Number of superdiagonals to keep. If negative, keep entire upper triangle.
    :return:
    """
    m, n = list(input.shape)[-2:]

    M = torch.zeros([m, n], dtype=torch.int32) + torch.arange(m, dtype=torch.int32).unsqueeze(1)
    N = torch.zeros([m, n], dtype=torch.int32) + torch.arange(n, dtype=torch.int32)

    in_band = (num_lower < 0 or (M - N) <= num_lower) == (num_upper < 0 or (N - M) <= num_upper)

    # band_part = torch.zeros_like(input).masked_scatter(in_band, input)
    band_part = input.masked_fill((in_band == 0), 0)

    return band_part


def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    """
    Implementation of sparse attention masks from Open AI's Sparse Transformer
    Paper: https://d4mucfpksywv.cloudfront.net/Sparse_Transformer/sparse_transformers.pdf
    Code: https://github.com/openai/sparse_attention
    :param int n: size of square matrix [1,1,n,n]
    :param attn_mode: attention mask shape
    :param local_attn_ctx: local attention context
    :return: torch.Tensor of shape [1, 1, n, n] with mask
    """

    if attn_mode == 'all':
        # mask with lower triangle
        m = torch.tril(torch.ones(n, n), diagonal=0)
    elif attn_mode == 'local':
        # mask which only attends to the last ctx inputs
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        m = matrix_band_part(torch.ones(n, n), ctx, 0)
    elif attn_mode == 'strided':
        stride = local_attn_ctx

        if stride < 1:
            raise ValueError(f"local_attn_ctx must be greater than 1 when using strided attention")

        # create square matricies which count rows(x) and columns(y)
        x = torch.arange(n, dtype=torch.int32).unsqueeze(1)
        y = torch.arange(n, dtype=torch.int32)

        zero = torch.zeros([n, n], dtype=torch.int32)

        q = zero + x
        k = zero + y

        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = c1 & c2

        m = c3.float()
    else:
        raise ValueError(f'Not yet implemented attn_mode: {attn_mode}')

    # reshape mask
    m = m.view((1, 1, n, n))
    return m


def show_attention(attention, title=None):
    """
    Display attention matrix or mask
    :param torch.Tensor attention: tensor of shape [n,m]
    """

    if title:
        plt.title(title)

    plt.imshow(attention)
    plt.show()


if __name__ == '__main__':

    # run visualisation
    for ctx in range(1, 10):
        local_mask = get_attn_mask(50, 'local', ctx)
        local_mask = local_mask.squeeze().squeeze()
        show_attention(local_mask, f'local_ctx: {ctx}')

    for stride in range(1, 10):
        stride_mask = get_attn_mask(50, 'strided', stride)
        stride_mask = stride_mask.squeeze().squeeze()
        show_attention(stride_mask, f'stride: {stride}')
