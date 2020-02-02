import torch
from torch import nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim)
    else:
        return nn.BatchNorm1d(dim)


class Conv1D(nn.Module):
    '''
       Build the Conv1D structure
       causal: if True is causal setting
    '''

    def __init__(self, in_channels=256, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Separation_TasNet, self).__init__()
        self.causal = causal
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.PReLu = nn.PReLU()
        self.norm = select_norm(norm, out_channels)
        self.pad = (dilation*(kernel_size-1)
                    )//2 if not causal else dilation*(kernel_size-1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                groups=out_channels, padding=self.pad, dilation=dilation)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, x):
        """
          Input:
              x: [B x C x T], B is batch size, T is times
          Returns:
              x: [B, C, T]
        """
        # B x C x T -> B x C_o x T_o
        x_conv = self.conv1x1(x)
        x_conv = self.PReLu(x_conv)
        x_conv = self.norm(x_conv)
        # B x C_o x T_o
        x_conv = self.dwconv(x_conv)
        x_conv = self.PReLu(x_conv)
        x_conv = self.norm(x_conv)
        # B x C_o x T_o -> B x C x T
        if self.causal:
            x_conv = x_conv[:, :, :-self.pad]
        x_conv = self.end_conv1x1(x_conv)
        return x+x_conv


class Encoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=kernel_size/2)

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Separation_TasNet(nn.Module):
    '''
       TasNet Separation part
       LayerNorm -> 1x1Conv -> 1-D Conv .... -> output
    '''

    def __init__(self, repeats=3, conv1d_block=8, in_channels=64, out_channels=128,
                 out_sp_channels=512, kernel_size=3, norm='gln', causal=False, num_spks=2):
        super(Separation_TasNet, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1d_list = self._sequential(
            repeats, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)
        self.PReLu = nn.PReLU()
        self.norm = select_norm('cln', in_channels)
        self.end_conv1x1 = nn.Conv1d(out_channels, num_spks*in_channels, 1)
        self.activation = nn.Sigmoid()
        sefl.num_spks = num_spks

    def _sequential(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_lists = [Conv1D(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return nn.Sequential(*Conv1D_lists)

    def forward(self, x):
        """
           Input:
               x: [B x C x T], B is batch size, T is times
           Returns:
               x: [B, C_o, T_o]
         """
        x = self.norm(x)
        # B x C x T
        x = self.conv1x1(x)
        # B x num_spks*N x T
        x = self.end_conv1x1(x)
        # num_spks x B x N x T
        x = torch.chunk(x, self.num_spks, dim=1)
        x = self.activation(torch.stack(x, dim=0))
        return x
