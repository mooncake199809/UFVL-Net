import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS, NORM_LAYERS

THRESHOLD = 0.5


@CONV_LAYERS.register_module()
class Conv2d_share(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, share_type=None,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
                        )
        self.share_type = 'channel_wise'
        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        if self.bias is not None:
            self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))

        # TODO: ablate one, zero and random
        # import pdb; pdb.set_trace()
        if self.share_type == 'channel_wise':
            self.score = nn.Parameter(torch.rand(self.weight.size(1)).cuda())
        elif self.share_type == 'kernel_wise':
            self.score = nn.Parameter(torch.rand((self.weight.size(2), self.weight.size(3))).cuda())
        else:
            self.score = nn.Parameter(torch.rand(1).cuda())

    def forward(self, input):
        if self.share_type == 'channel_wise':
            score = binarizer_fn(self.score).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif self.share_type == 'kernel_wise':
            score = binarizer_fn(self.score).unsqueeze(0).unsqueeze(0)
        else:
            score = binarizer_fn(self.score)

        weight = score * self.weight + (1 - score) * self.specific_weight
        if self.bias is not None:
            bias = score * self.bias + (1 - score) * self.specific_bias
        else:
            bias = None
        return self._conv_forward(input, weight)


@NORM_LAYERS.register_module()
class BatchNorm2d_share(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        if self.bias is not None:
            self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))
        self.score = nn.Parameter(torch.rand(1).cuda())

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        score = binarizer_fn(self.score)

        weight = score * self.weight + (1 - score) * self.specific_weight
        if self.bias is not None:
            bias = score * self.bias + (1 - score) * self.specific_bias
        else:
            bias = None

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class BinarizerFn(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold=THRESHOLD):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 1
        outputs[inputs.gt(threshold)] = 0
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput, None


binarizer_fn = BinarizerFn.apply