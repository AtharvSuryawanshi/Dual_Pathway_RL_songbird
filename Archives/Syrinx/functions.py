import numpy as np
import os
from scipy.interpolate import interp2d

# save plots in a folder
save_dir = "plots"
def remove_prev_files(): # 
    '''removes previous files in the directory'''
    os.makedirs(save_dir, exist_ok = True)
    for filename in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, filename))

# Basic functions
def gaussian(coordinates, height, mean, spread):
    constant = 1 / (2 * spread**2)  # Pre-compute if spread is constant
    x, y = coordinates[0], coordinates[1]
    diff_x = x - mean[0]
    diff_y = y - mean[1]
    return height * np.exp(-constant * (diff_x**2 + diff_y**2))

def new_sigmoid(x, m=0.0, a=0.0):
    if m != 0.0:  # Check if m is not zero (avoid division by zero)
        constant = -m * a
    exp_term = np.exp(constant + (-m * x))
    return (2 / (1 + exp_term)) - 1

def sigmoid(x, m =0.0 , a=0.0 ):
    """ Returns an output between 0 and 1 """
    return 1 / (1 + np.exp(-1*(x-a)*m))

def sym_lognormal_samples(minimum, maximum, size, mu = 0.01, sigma = 0.5):
    """
    This function generates samples from a combined (original + reflected) lognormal distribution.
    Args:
        mu (float): Mean of the underlying normal distribution.
        sigma (float): Standard deviation of the underlying normal distribution.
        size (int): Number of samples to generate.
    Returns:
        numpy.ndarray: Array of samples from the combined lognormal distribution.
    """
    if size == 0:
        ValueError('Size cannot be zero')
    # Generate lognormal samples with half in one dimension only
    samples = np.random.lognormal(mu, sigma, size)
    combined_samples = np.concatenate((samples, samples * -1))/4
    # randomly remove samples such that size of combined_samples is equal to size
    combined_samples = np.random.choice(combined_samples.reshape(-1), size, replace = False)
    combined_samples = np.clip(combined_samples, minimum, maximum)
    return combined_samples

def lognormal_weight(size, mu = 0.01, sigma = 0.5):
    '''returns lognormal weights'''
    samples = np.random.lognormal(mu, sigma, size)/4
    samples = np.clip(samples, 0, 1)
    return samples

def make_contour(Z, n = 256):
    Z = Z / Z.max()

    # Transform exponentially
    Z = np.power(1000, Z)
    Z = Z / Z.max()

    # Interpolate
    x = np.linspace(0, 1., Z.shape[0])
    y = np.linspace(0, .2, Z.shape[1])

    x2 = np.linspace(0, 1., n)
    y2 = np.linspace(0, .2, n)
    f = interp2d(x, y, Z, kind='cubic')
    Z = f(x2, y2)

    Z = (Z-np.min(Z))/(np.max(Z)-np.min(Z))
    Z = Z / Z.max()

    targetZpos = np.argwhere(Z==1)[0]
    targetpos = np.zeros((2))
    targetpos[0] = (targetZpos[1] / Z.shape[1]) * 2 - 1
    targetpos[1] = (targetZpos[0] / Z.shape[0]) * 2 - 1
    return Z, targetpos