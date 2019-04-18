import numpy as np
from scipy.spatial.distance import pdist, squareform
from abc import abstractmethod


class Kernel(object):

    """
    Abstract class for implementing kernels
    """

    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self.K = None

    @abstractmethod
    def compute_kernel(self, X, Y):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class GaussianKernel(Kernel):

    """
    Gaussian kernel for MMD tests
    """

    def __init__(self, sigma=1):
        super().__init__(kernel_type="Gaussian")
        self.sigma = sigma
        self.K = 1  # always 1 for Gaussian kernel, on the diagonal

    def compute_kernel(self, X, Y):
        """
        Vectorized computation of gaussian kernel exp(-||x-y||^2_2 / (2*(sigma^2)))

        :param g1: first set of samples, of shape (n_features, m)
        :param g2: second set of samples, of shape (n_features, n)
        :param sigma: std parameter in gaussian kernel
        :return: kernel matrix containing kernel on
            - g1 (kernel[:m, :m]),
            - g2 (kernel[m:, m:]),
            - and kernel between g1 and g2 (kernel[m:, :m] which is equal to kernel[:m, m:].T)

        Verification :
            m = g1.shape[1]
            n = g2.shape[1]
            i = 1  # index on g1
            j = 2  # index on g2
            assert (i >= 0) and (i < m)
            assert (j >= 0) and (j < n)
            assert np.exp(-np.sum((g1.T[i] - g2.T[j]) ** 2) / (sigma ** 2)) == pd.DataFrame(K).iloc[i, m+j]
        """
        XY = np.vstack([X.T, Y.T])
        pairwise_sq_dists = squareform(pdist(XY, "sqeuclidean"))
        kernel = np.exp(-pairwise_sq_dists / (2 * (self.sigma ** 2)))
        return kernel

    def __repr__(self):
        return f"Kernel: Gaussian Kernel with sigma={self.sigma}"
