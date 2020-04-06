# Utilities for Neural Network training
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

def custom_initializer(width_nn,wgts_array):
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
    mean = 0.1
    n_random_wgts = width_nn - wgts_array.shape[0]
    wgts_array = np.append(wgts_array,np.random.normal(loc = mean, scale = mean, size = n_random_wgts))
    return tf.constant_initializer(wgts_array)

class CustomDense(tf.keras.layers.Dense):

        def __init__(self, wgts_array, custom_wgt = False, **kwargs):
            if custom_wgt == True:
                super(CustomDense,self).__init__(kernel_initializer = custom_initializer(width,wgts_array),**kwargs)
            else:
                super(CustomDense,self).__init__(kernel_initializer = "glorot_uniform",**kwargs)


