import datetime

from pytz import timezone


def local_time(fmt="%Y-%m-%d_%H:%M", local_tz='Australia/Brisbane'):
    """
    Get current local time
    :param fmt:
    :param tz:
    :return:
    """
    time = datetime.datetime.now(timezone('UTC'))
    time = time.astimezone(timezone(local_tz))
    time = time.strftime(fmt)
    return time


def count_model_param(net, trainable_only=True, print_info=True):
    """
    Return the number of parameters in a model
    :param net: pytorch nn.Module
    :param trainable_only: (bool) to return number of trainable values
    :return: Number of parameters in the pytorch model
    """
    if trainable_only:
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in net.parameters())

    if print_info:
        print(f'     Trainable Parameters: {params / 1000000} million')

    return params
