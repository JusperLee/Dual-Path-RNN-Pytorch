import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils.util import check_parameters

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
        super(Conv1D, self).__init__()
        self.causal = causal
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.PReLu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)
        self.pad = (dilation*(kernel_size-1)
                    )//2 if not causal else dilation*(kernel_size-1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLu2 = nn.PReLU()
        self.norm2 = select_norm(norm,out_channels)
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
        x_conv = self.PReLu1(x_conv)
        x_conv = self.norm1(x_conv)
        # B x C_o x T_o
        x_conv = self.dwconv(x_conv)
        x_conv = self.PReLu2(x_conv)
        x_conv = self.norm2(x_conv)
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
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                kernel_size=kernel_size, stride=kernel_size//2,groups=1)

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

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x,dim=1)
        else:
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
        self.conv1d_list = self._Sequential(
            repeats, conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)
        self.PReLu = nn.PReLU()
        self.norm = select_norm('cln', in_channels)
        self.end_conv1x1 = nn.Conv1d(out_channels, num_spks*in_channels, 1)
        self.activation = nn.Sigmoid()
        self.num_spks = num_spks

    def _Sequential_block(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_lists = [Conv1D(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return nn.Sequential(*Conv1D_lists)
    
    def _Sequential(self, num_repeats, num_blocks, **block_kwargs):
        '''
           Sequential repeats
           input:
                 num_repeats: Number of repeats
                 num_blocks: Number of block in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        repeats_lists = [self._Sequential_block(
            num_blocks, **block_kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeats_lists)

    def forward(self, x):
        """
           Input:
               x: [B x C x T], B is batch size, T is times
           Returns:
               x: [num_spks, B, N, T]
         """
        # B x C x T
        x = self.norm(x)
        x = self.conv1x1(x)
        # B x C x T
        x = self.conv1d_list(x)
        # B x num_spks*N x T
        x = self.PReLu(x)
        x = self.end_conv1x1(x)
        # num_spks x B x N x T
        x = torch.chunk(x, self.num_spks, dim=1)
        x = self.activation(torch.stack(x, dim=0))
        return x


class Conv_TasNet(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''

    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 norm="gln",
                 num_spks=2,
                 activate="relu",
                 causal=False):
        super(Conv_TasNet, self).__init__()
        self.encoder = Encoder(kernel_size=L, out_channels=N)
        self.separation = Separation_TasNet(repeats=R, conv1d_block=X, in_channels=N,
                                            out_channels=B, out_sp_channels=H, kernel_size=P,
                                            norm=norm, causal=causal, num_spks=num_spks)
        self.decoder = Decoder(
            in_channels=N, out_channels=1, kernel_size=L, stride=L//2)
        self.num_spks = num_spks

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, T]
        """
        # B x T -> B x C x T
        x_encoder = self.encoder(x)
        # B x C x T -> num_spks x B x N x T
        x_sep = self.separation(x_encoder)
        # [B x N x T, B x N x T]
        audio_encoder = [x_encoder*x_sep[i] for i in range(self.num_spks)]
        # [B x T, B x T]
        audio = [self.decoder(audio_encoder[i]) for i in range(self.num_spks)]
        return audio


if __name__ == "__main__":
    conv = Conv_TasNet()
    #encoder = Encoder(16, 512)
    x = torch.randn(4, 32000)
    out = conv(x)
    print("{:.3f}".format(check_parameters(conv)))
