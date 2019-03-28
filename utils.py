import torch


def compute_pairwise_distances(x, y):
    """
        Computes the squared pairwise Euclidean distances between x and y.
        Args:
            x: a tensor of shape [num_x_samples, num_features]
            y: a tensor of shape [num_y_samples, num_features]
        Returns:
            a distance matrix of dimensions [num_x_samples, num_y_samples].
        Raises:
            ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.size()[1] != y.size()[1]:
        raise ValueError('The number of features should be the same.')

    x = torch.unsqueeze(x, dim=2)  # (num_x_samples, num_features, 1)
    y = torch.transpose(y, dim0=0, dim1=1)  # (num_features, num_y_samples)
    x_minus_y = x - y  # (num_x_samples, num_features, num_y_samples)
    dist = torch.sum(torch.mul(x_minus_y, x_minus_y), dim=1)  # (num_x_samples, num_y_samples)
    # dist = torch.transpose(dist, dim0=0, dim1=1)  # (num_y_samples, num_x_samples)
    return dist


def gaussian_kernel_matrix(x, y, sigmas=None):
    """
        Computes a Guassian Radial Basis Kernel between the samples of x and y.
        We create a sum of multiple gaussian kernels each having a width sigma_i.
        Args:
            x: a tensor of shape [num_x_samples, num_features]
            y: a tensor of shape [num_y_samples, num_features]
            sigmas: a tensor of floats which denote the widths of each of the
            gaussians in the kernel.
        Returns:
            A tensor of shape [num_x_samples, num_y_samples] with the RBF kernel.
    """
    if sigmas is None:
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    
    beta = 1. / (2. * (torch.unsqueeze(torch.FloatTensor(sigmas), dim=1)))

    dist = compute_pairwise_distances(x, y)  # (num_x_samples, num_y_samples)
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))  # (n, num_x_samples*num_y_samples)
    s = torch.sum(torch.exp(-s), dim=0)  # (num_x_samples*num_y_samples)
    out = torch.reshape(s, dist.size())  # (num_x_samples, num_y_samples)
    return out


def MMD(x, y):
    matx = gaussian_kernel_matrix(x, x)  # (num_x_samples, num_x_samples)
    maty = gaussian_kernel_matrix(y, y)  # (num_y_samples, num_y_samples)
    matxy = gaussian_kernel_matrix(x, y)  # (num_x_samples, num_y_samples)
    mmd = torch.mean(matx) + torch.mean(maty) - 2 * torch.mean(matxy)
    mmd = torch.where(mmd>0, mmd, torch.zeros((mmd.size())))
    return mmd
