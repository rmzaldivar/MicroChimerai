import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from einops import rearrange
from random import choice
from itertools import chain

class HyenaOperator(nn.Module):
    """
    Hyena operator for efficient sequence processing and long-range dependencies.
    """

    def __init__(self, d_model, L, H, channels, dropout=0.1,
                 kernel_learning_rate=0.01, kernel_lam=0.1, kernel_dropout=0.1):
        """
        Initialize the HyenaOperator.

        :param d_model: Input and output dimensions (int)
        :param L: Maximum input sequence length (int)
        :param H: Width of the FFN (int)
        :param channels: Number of channels in the input (int)
        :param dropout: Dropout probability (float)
        :param kernel_learning_rate: Learning rate of the kernel (float)
        :param kernel_lam: Regularization parameter for the kernel (float)
        :param kernel_dropout: Dropout probability for the kernel (float)
        """
        super().__init__()
        self.H = H
        self.L = L * 2  # for causal conv
        self.channels = channels
        self.dropout = nn.Dropout(p=dropout)
        self.kernel_learning_rate = kernel_learning_rate
        self.kernel_lam = kernel_lam
        self.kernel_drop = torch.nn.Dropout(p=kernel_dropout)

        self.D = nn.Parameter(torch.randn(channels, self.H))

        self.activation = nn.GELU()

        self.output_linear = nn.Sequential(
            nn.Linear(self.channels * self.H, 2 * self.H, bias=True),
            nn.GLU(dim=-1),
        )

        self.kernel = torch.nn.Parameter(torch.randn(self.channels, self.H, self.L) * 0.002)

        self.register("kernel", self.kernel, kernel_learning_rate)

    def forward(self, u):
        L = u.size(-1)

        k = self.kernel

        # squash operator
        k = F.relu(torch.abs(k) - self.kernel_lam) * torch.sign(k)
        k = self.kernel_drop(k)

        # use FFT to compute convolution
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]

        # Compute skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        # Transpose for the linear
        y = y.transpose(-1, -2)
        y = self.output_linear(y)
        y = y.transpose(-1, -2)

        return y