# Utilities for Neural Network training
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

def get_number_units(parameters, input_size = None, output_size = None):
    units  = (parameters-output_size)/(input_size + output_size + 1)
    return units

def custom_bias(units,trained_wgts):
    """Custom Initializer to create partially random weights
    from smaller trained weights
    
    Args:
        width_nn (int): Width of the Dense layer of the NN 
        wgts_array (ndarray): Trained Weights

    Returns:
        tf.initializer : Custom initilizer with custom weights
    """    
    #Parameters for the random Normal Weights
    mean = 0
    std_dev = 0.1
    n_random_wgts = units - trained_wgts.shape[0]
    random_wgts = tf.random.normal(mean = mean, stddev = std_dev, shape = [n_random_wgts])
    new_wgts = np.concatenate([trained_wgts,random_wgts])
    return new_wgts

def custom_kernel(units,trained_wgts):
    """Custom Initializer to create partially random weights
    from smaller trained weights
    
    Args:
        width_nn (int): Width of the Dense layer of the NN 
        wgts_array (ndarray): Trained Weights

    Returns:
        np.array : Custom initilizer with custom weights
    """    
    #Parameters for the random Normal Weights
    mean = 0
    std_dev = 0.1
    n_random_wgts = units - trained_wgts.shape[1]
    random_wgts = tf.random.normal(mean = mean, stddev = std_dev, shape = (784,n_random_wgts))
    new_wgts = np.concatenate([trained_wgts,random_wgts], axis =1)
    return new_wgts

