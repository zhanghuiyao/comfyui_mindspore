from mindspore import nn, mint


class PixelNorm(nn.Cell):
    def __init__(self, dim=1, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    def construct(self, x):
        return x / mint.sqrt(mint.mean(x**2, dim=self.dim, keepdim=True) + self.eps)
