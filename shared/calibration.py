import numpy as np

def temperature_scale(logit, T=1.5):
    return 1 / (1 + np.exp(-logit / T))



