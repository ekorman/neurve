import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


def resnet_init(m):
    """Layer initialization used by ResNet
    (https://github.com/pytorch/vision/blob/27278ec8887a511bd7d6f1202d50b0da7537fc3d/
    torchvision/models/resnet.py#L160)

    Parameters
    ----------
    m : nn.Module
        layer to initialize
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def modify_resnet_model(model, in_channels, dim_z):
    """Modifies some layers of a given torchvision resnet model to
    match the one used for the CIFAR-10 experiments in the SimCLR paper.

    Modified from https://github.com/Spijkervet/SimCLR/blob/master/modules/resnet_hacks.py
    by also adding the ability to change the output dimension of the last
    bottleneck.

    Parameters
    ----------
    model : ResNet
        Instance of a torchvision ResNet model.
    in_channels : int
        number of input channels (1 or 3)
    dim_z : int
        desired dimension of the output of the model.

    Returns
    -------
    nn.Module
        Modified ResNet model.
    """
    assert isinstance(model, ResNet), "model must be a ResNet instance"

    conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    resnet_init(conv1)
    model.conv1 = conv1
    model.maxpool = nn.Identity()

    for i in range(2, 5):
        layer = getattr(model, "layer{}".format(i))
        block = list(layer.children())[0]
        if isinstance(block, Bottleneck):
            assert block.conv1.kernel_size == (
                1,
                1,
            ) and block.conv1.stride == (
                1,
                1,
            )
            assert block.conv2.kernel_size == (
                3,
                3,
            ) and block.conv2.stride == (
                2,
                2,
            )
            assert block.conv2.dilation == (
                1,
                1,
            ), "Currently, only models with dilation=1 are supported"
            block.conv1.stride = (2, 2)
            block.conv2.stride = (1, 1)

    out_dim = model.fc.in_features
    model.fc = nn.Identity()
    if out_dim == dim_z:
        return model

    return nn.Sequential(model, nn.Linear(out_dim, dim_z))
