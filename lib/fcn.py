# import warnings
from torch.nn import Module
from torch.nn import Conv1d
# from torch.nn import ConvTranspose1d
from torch.nn import Conv2d
# from torch.nn import ConvTranspose2d
from torch.nn import Dropout
from torch.nn import BatchNorm1d
from torch.nn import Upsample
# from torch.nn import MaxPool1d
from torch.nn.functional import relu


class FCN(Module):
    def __init__(self, input_dims, using_batch_norm, using_pooling,
                 pooling_method, conv1_kernel_dims, conv1_num_kernels,
                 conv1_stride_dims, conv1_dropout, pool1_kernel_dims,
                 pool1_stride,

                       conv2_kernel_dims,
                       conv2_num_kernels,
                       conv2_stride_dims,
                       conv2_dropout,

                       pool2_kernel_size,
                       pool2_stride,

                       conv3_kernel_dims,
                       conv3_num_kernels,
                       conv3_stride_dims,
                       conv3_dropout):


        super(FCN, self).__init__()

        input_height, input_width, input_depth = input_dims

        # Validate input dims
        assert input_height in [1, 2]
        assert input_width in [65, 130]
        assert input_depth in [1, 2]

        mode = '1D' if input_height == 1 else '2D'

        # Instance attributes for use in self.forward() later.
        self.input_dims = input_dims
        self.using_batch_norm = using_batch_norm

        conv1_kernel_height, conv1_kernel_width = conv1_kernel_dims
        conv1_stride_height, conv1_stride_width = conv1_stride_dims

        # Conv1 output size is in [H, W, D]
        conv1_output_height = (input_height - conv1_kernel_height) / conv1_stride_height + 1
        conv1_output_width = (input_width - conv1_kernel_width) / conv1_stride_width + 1
        conv1_output_depth = conv1_num_kernels
        conv1_output_size = [conv1_output_height, conv1_output_width, conv1_output_depth]

        if mode == '1D':
            self.conv1 = Conv1d(input_depth, conv1_output_depth, conv1_kernel_dims, stride=conv1_stride_dims)
        if mode == '2D':
            self.conv1 = Conv2d(input_depth, conv1_output_depth, conv1_kernel_dims, stride=conv1_stride_dims)

        self.conv1_drop = Dropout(p=conv1_dropout)
        if using_batch_norm is True:
            self.batch_norm1 = BatchNorm1d(conv1_num_kernels)

        # Conv2
        conv2_kernel_height, conv2_kernel_width = conv2_kernel_dims
        conv2_stride_height, conv2_stride_width = conv2_stride_dims

        conv2_output_height = (conv1_output_height - conv2_kernel_height) / conv2_stride_height + 1
        conv2_output_width = (conv1_output_width - conv2_kernel_width) / conv2_stride_width + 1
        conv2_output_depth = conv2_num_kernels
        conv2_output_size = [conv2_output_height, conv2_output_width, conv2_output_depth]

        if mode == '1D':
            self.conv2 = Conv1d(conv1_output_depth, conv2_output_depth, conv2_kernel_dims, stride=conv2_stride_dims)
        if mode == '2D':
            self.conv2 = Conv2d(conv1_output_depth, conv2_output_depth, conv2_kernel_dims, stride=conv2_stride_dims)

        self.conv2_drop = Dropout(p=conv2_dropout)
        if using_batch_norm is True:
            self.batch_norm2 = BatchNorm1d(conv2_num_kernels)

        # Upsample layer
        self.upsample = Upsample(scale_factor=2)

        # Conv3
        conv3_kernel_height, conv3_kernel_width = conv3_kernel_dims
        conv3_stride_height, conv3_stride_width = conv3_stride_dims

        conv3_output_height = (conv2_output_height - conv3_kernel_height) / conv3_stride_height + 1
        conv3_output_width = (conv2_output_width - conv3_kernel_width) / conv3_stride_width + 1
        conv3_output_depth = conv3_num_kernels
        conv3_output_size = [conv3_output_height, conv3_output_width, conv3_output_depth]

        if mode == '1D':
            self.conv3 = Conv1d(conv2_output_depth, input_depth, conv3_kernel_dims, stride=conv3_stride_dims)
        if mode == '2D':
            self.conv3 = Conv2d(conv2_output_depth, input_depth, conv3_kernel_dims, stride=conv3_stride_dims)

        # if mode == '1D':
        #     self.conv2 = ConvTranspose1d(conv2_output_depth, input_depth, conv2_kernel_dims, stride=conv2_stride_dims)
        # if mode == '2D':
        #     self.conv2 = ConvTranspose2d(conv2_output_depth, input_depth, conv2_kernel_dims, stride=conv2_stride_dims)

    def forward(self, x): # pylint: disable=W0221
        x = x.view(-1, *self.input_dims)
        x = relu(self.conv1(x))
        if self.batch_norm is True:
            x = self.batch_norm1(x)

        x = relu(self.conv2(x))
        if self.batch_norm is True:
            x = self.batch_norm2(x)

        x = self.upsample(x)
        x = relu(self.conv3(x))

        return x
