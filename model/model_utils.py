import torch
import torch.nn as nn
import torch.nn.functional as nnf


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class BnReluConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch'):
        super(BnReluConv, self).__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm3d(in_channels)
        else:
            self.norm = nn.InstanceNorm3d(in_channels)
        if activate == 'relu':
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch'):
        super(ResidualBlock, self).__init__()
        self.bn_relu_conv1 = BnReluConv(in_channels, out_channels, stride, kernel_size, padding, activate, norm)
        self.bn_relu_conv2 = BnReluConv(out_channels, out_channels, stride, kernel_size, padding, activate, norm)

    def forward(self, x):
        y = self.bn_relu_conv1(x)
        residual = y
        z = self.bn_relu_conv2(y)
        return z + residual


class DeResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch'):
        super(DeResidualBlock, self).__init__()
        self.bn_relu_conv1 = BnReluConv(in_channels, out_channels, stride, kernel_size, padding, activate, norm)
        self.bn_relu_conv2 = BnReluConv(out_channels, out_channels, stride, kernel_size, padding, activate, norm)

    def forward(self, x1, x2):
        y = self.bn_relu_conv1(x1)
        y = self.bn_relu_conv2(y)
        return y + x2


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, activate='relu', norm='batch'):
        super(UpConv, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        if norm == 'batch':
            self.norm = nn.BatchNorm3d(in_channels)
        else:
            self.norm = nn.InstanceNorm3d(in_channels)
        if activate == 'relu':
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

