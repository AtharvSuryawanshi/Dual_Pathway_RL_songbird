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

def running_mean(x, N=5):
        """ Returns the running average of an array. """
        rm = np.convolve(x, np.ones(N)/N, mode='valid')
        padded_rm = np.ones(np.shape(x)) * rm[-1]
        padded_rm[:rm.size] = rm

        return padded_rm

def running_mean_dynamic(x, N_i=3, N_f=50):
    """
    Running mean with linearly increasing window size.
    
    Parameters
    ----------
    x : array-like
        Input array.
    N_i : int
        Initial window size at the start.
    N_f : int
        Final window size at the end.
        
    Returns
    -------
    smoothed : np.ndarray
        Smoothed array of same length as x.
    """
    x = np.asarray(x)
    n = len(x)
    smoothed = np.zeros(n)

    # Window sizes increase linearly from N_i to N_f
    window_sizes = np.linspace(N_i, N_f, n).astype(int)

    for i in range(n):
        N = window_sizes[i]
        start = max(0, i - N//2)
        end = min(n, i + N//2 + 1)
        smoothed[i] = np.mean(x[start:end])
    
    return smoothed

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

def find_neighboring_directories():
    """
    Finds all directories (folders, except pycache) in the same directory as the currently running Python script.

    Returns:
        list: A list of directory names found in the same directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_entries = os.listdir(current_dir)
    directories = []
    for entry in all_entries:
        if entry != "__pycache__":  # Skip the cache directory
            full_path = os.path.join(current_dir, entry)  # Get full path for entry
            if os.path.isdir(full_path):  # Check if it's a directory
                if entry != "__pycache__" and entry != "plots": 
                    directories.append(entry)
    return directories

from scipy.interpolate import RegularGridInterpolator

def make_contour(Z, n=256):
    Z = Z / Z.max()

    # Transform exponentially
    Z = np.power(1000, Z)
    Z = Z / Z.max()

    # Interpolate
    x = np.linspace(0, 1., Z.shape[0])
    y = np.linspace(0, .2, Z.shape[1])

    x2 = np.linspace(0, 1., n)
    y2 = np.linspace(0, .2, n)
    
    interpolating_function = RegularGridInterpolator((x, y), Z, method='cubic')
    x2_grid, y2_grid = np.meshgrid(x2, y2)
    points = np.array([x2_grid.flatten(), y2_grid.flatten()]).T
    Z = interpolating_function(points).reshape(n, n)

    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z)) # type: ignore
    Z = Z / Z.max()

    targetZpos = np.argwhere(Z == 1)[0]
    targetpos = np.zeros((2))
    targetpos[0] = (targetZpos[1] / Z.shape[1]) * 2 - 1
    targetpos[1] = (targetZpos[0] / Z.shape[0]) * 2 - 1
    return Z, targetpos

# def make_contour(Z, n = 256):
#     Z = Z / Z.max()

#     # Transform exponentially
#     Z = np.power(1000, Z)
#     Z = Z / Z.max()

#     # Interpolate
#     x = np.linspace(0, 1., Z.shape[0])
#     y = np.linspace(0, .2, Z.shape[1])

#     x2 = np.linspace(0, 1., n)
#     y2 = np.linspace(0, .2, n)
#     f = interp2d(x, y, Z, kind='cubic')
#     Z = f(x2, y2) 

#     Z = (Z-np.min(Z))/(np.max(Z)-np.min(Z))
#     Z = Z / Z.max()

#     targetZpos = np.argwhere(Z==1)[0]
#     targetpos = np.zeros((2))
#     targetpos[0] = (targetZpos[1] / Z.shape[1]) * 2 - 1
#     targetpos[1] = (targetZpos[0] / Z.shape[0]) * 2 - 1
#     return Z, targetpos
