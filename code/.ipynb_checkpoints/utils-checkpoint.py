# Utilities for Neural Network training
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

class Training:
    
    def __init__(self, parameters, network):
        self.units = parameters
        self.network = network
    
    def get_network(self):
        return self.network
    
    def compile(self,**kwargs):
        self.network.compile(**kwargs)
        
    def fit(self,**kwargs):
        self.network.fit(**kwargs)
        
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


class CustomDense(keras.layers.Dense):

        def __init__(self, units, wgts_array=None, custom_wgt = False, **kwargs):
            if custom_wgt == True:
                super(CustomDense,self).__init__(units, kernel_initializer = custom_initializer(units,wgts_array), **kwargs)
            else:
                super(CustomDense,self).__init__(units, kernel_initializer = "glorot_uniform", **kwargs)
        
        def call(self,input):
            x = super().call(input)
            return x


class CustomNN(keras.Model):
    def __init__(self, units, num_classes, wgts_array=None, custom_wgt = False, **kwargs):
        super(CustomNN,self).__init__()
        self.flatten = keras.layers.Flatten(input_shape = (28, 28))
        self.custom_dense = CustomDense(units = units,
                                        wgts_array = wgts_array , custom_wgt = custom_wgt,
                                        **kwargs)
        self.output_layer = keras.layers.Dense(units = num_classes)
        
    def call(self,x):
        x = self.flatten(x)
        x = self.custom_dense(x)
        return self.output_layer(x)
