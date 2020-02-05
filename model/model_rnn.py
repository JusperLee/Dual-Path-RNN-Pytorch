import torch.nn.functional as F
from torch import nn
import torch
from utils.util import check_parameters
import sys
sys.path.append('../')


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
        self.norm2 = select_norm(norm, out_channels)
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
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1)

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


class Decoder(nn.ConvTranspose2d):
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
        x: N x K x S or N x C x K x S
        """
        if x.dim() not in [3, 4]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 4 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 2:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, in_channels, hidden_channels, rnn_type='LSTM', norm='ln', dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        self.rnn = getattr(nn, rnn_type)(
            in_channels, hidden_channels, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.norm = select_norm(norm, in_channels)
        self.linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, in_channels)
        self.conv2d = nn.Conv2d(in_channels,in_channels*num_spks,kernel_size=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.linear(intra_rnn)
        # [BS, N, K]
        intra_rnn = intra_rnn.permute(0, 2, 1).contiguous()
        intra_rnn = self.norm(intra_rnn)
        # [B, S, N, K]
        intra_rnn = intra_rnn.view(B, S, N, K)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous()
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.linear(inter_rnn)
        # [BK, N, S]
        inter_rnn = inter_rnn.permute(0, 2, 1).contiguous()
        inter_rnn = self.norm(inter_rnn)
        # [B, K, N, S]
        inter_rnn = inter_rnn.view(B, K, N, S)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 2, 1, 3).contiguous()
        out = inter_rnn + intra_rnn

        # [B, N*spks, K, S]
        out = self.prelu(out)
        out = self.conv2d(out)
        return out


class Dual_Path_RNN(nn.Module):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
    '''

    def __init__(self, in_channels, hidden_channels,
                 rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.dual_rnn = self._Sequential(in_channels, hidden_channels,
                                         rnn_type='LSTM', norm='ln', dropout=0,
                                         bidirectional=False, num_layers=4)
        
        self.conv1d = 

    def _Sequential(self, in_channels, hidden_channels,
                    rnn_type='LSTM', norm='ln', dropout=0,
                    bidirectional=False, num_layers=4):
        block_list = [Dual_RNN_Block(in_channels, hidden_channels,
                                     rnn_type='LSTM', norm='ln', dropout=0,
                                     bidirectional=False) for i in range(num_layers)]
        return nn.Sequential(*block_list)

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape()
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            input = torch.cat((input, torch.zeros(B, N, gap)), dim=2)

        pad = torch.zeros(B, N, P)
        input = torch.cat((pad, input, pad), dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape()
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat((input1, input2), dim=3).view(
            B, N, -1, K).permute(0, 1, 3, 2).contiguous()

        return input, gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
        '''
        B, N, K, S = input.shape()
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


if __name__ == "__main__":
    rnn = Dual_RNN(25, 100)
    #encoder = Encoder(16, 512)
    x = torch.randn(10, 25, 9, 40)
    out = rnn(x)
    print("{:.3f}".format(check_parameters(rnn)))
    print(out.shape)
