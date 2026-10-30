import numpy as np
from functions import *

class NN:
    def __init__(self, parameters, seed):
        np.random.seed(seed)
        self.out_size = parameters['const']['OUT_SIZE']
        
    def forward(self, action, noise, parameters, iter):
        return np.clip(action + np.random.normal(0, noise, self.out_size), -1.5, 1.5)