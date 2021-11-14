import numpy as np


class BatchNormNumpy():
    def __init__(self):
        self.x_normalized = None
        self.bn_param = {'mode': 'train'}
        self.cache = None

    def batchnorm_forward(self, x, gamma, beta):
        """
        Forward pass for batch normalization.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean of features
          - running_var Array of shape (D,) giving running variance of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = self.bn_param['mode']
        eps = self.bn_param.get('eps', 1e-5)
        momentum = self.bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = self.bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = self.bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

        out, self.cache = None, None
        if mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)

            # Normalization followed by Affine transformation
            self.x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
            out = gamma * self.x_normalized + beta

            # Estimate running average of mean and variance to use at test time
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

            # Cache variables needed during backpropagation
            self.cache = (x, sample_mean, sample_var, gamma, beta, eps)

        elif mode == 'test':
            # normalize using running average
            x_normalized = (x - running_mean) / np.sqrt(running_var + eps)

            # Learned affine transformation
            out = gamma * x_normalized + beta

        # Store the updated running means back into bn_param
        self.bn_param['running_mean'] = running_mean
        self.bn_param['running_var'] = running_var

        return out

    def batchnorm_backward(self, dout):
        """
        Backward pass for batch normalization.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
        """
        # Unpack cache variables
        x, sample_mean, sample_var, gamma, beta, eps = self.cache

        # See derivations above for dgamma, dbeta and dx
        dgamma = np.sum(dout * self.x_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)

        m = x.shape[0]
        t = 1. / np.sqrt(sample_var + eps)

        dx = (gamma * t / m) * (m * dout - np.sum(dout, axis=0)
                                - t ** 2 * (x - sample_mean) * np.sum(dout * (x - sample_mean), axis=0))

        return dx, dgamma, dbeta


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    N, D = 2, 5
    x = 5 * np.random.randn(N, D) + 12
    # gamma = np.random.randn(D)
    gamma = np.ones(D)
    # beta = np.random.randn(D)
    beta = np.zeros(D)
    dout = np.random.randn(N, D)

    bn = BatchNormNumpy()
    bn.bn_param['momentum'] = 0.0
    out = bn.batchnorm_forward(x, gamma, beta)
    dx1, dgamma1, dbeta1 = bn.batchnorm_backward(dout)

    print(out)

    bn_torch = nn.BatchNorm1d(D, affine=True, momentum=.5)
    print('weight')
    print(bn_torch.weight)

    out_torch = bn_torch(torch.from_numpy(x).float())
    print(out_torch)

    m = nn.BatchNorm1d(5)
    # m = nn.BatchNorm1d(5, affine=False)
    input = torch.randn(2, 5)
    output = m(input)
    print(output)
    # input2 = torch.randn(2, 5)
    # output = m(input2)
    # print(output)

