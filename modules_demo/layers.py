import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomUniform, RandomNormal
import numpy as np
from tensorflow import convert_to_tensor

    
class ProbMask(Layer):
    
    def __init__(self, 
                 slope=10,
                 initializer=None,
                 trainable = True,
                 **kwargs):
        """
        The layer of the mask containing the trainable parameters.
        Such parameters go through a sigmoid and generate the prob_mask.
        Agrs:
            slope: Float, the steepness of the sigmoid. It is not 
                an "important" parameter, 5 is a standard value.
            initializer: tf.keras.Initializer, how the parameters are
                initialized. If None, prob_mask is uniformly distributed
                in [0,1].
            trainable: Bool, whether to set trainable or not the mask.
        """
        
        with tf.init_scope():
            if initializer == None:
                self.initializer = self._logit_slope_random_uniform
            else:
                self.initializer = initializer

        self.slope = slope
        self.trainable = trainable
        super(ProbMask, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'slope':self.slope,
            'initializer':self.initializer,
            'trainable':self.trainable,
        })
        return config

    def build(self, input_shape):

        # The only trainable parameters of the mask
        with tf.init_scope():
        
            self.mult = self.add_weight(
                name = 'logit_weight',
                shape=input_shape[1:],  
                initializer=self.initializer,  
                trainable=self.trainable,
               )
        
        super(ProbMask, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self,x):
        
        # multiply the input with 0 and adding it to the mask allows
        # to maintain the correct shape ([batch_size, ...])
        prob_mask = 0*x + tf.sigmoid(self.slope * self.mult)
        
        return prob_mask

    def compute_output_shape(self, input_shape):
        
        return input_shape
    
    def read_probMask(self, apply_sigmoid = True, ):
        
        """
        Returns the prob_mask
        Args:
            apply_sigmoid: Bool, whether to return the prob_mask in [0,1] or
                the trainable tensor before the sigmoid is applied in R.
        Returns:
            The prob_mask or the parameters generating the prob_mask
        """
        
        prob_mask = self.mult
        
        if apply_sigmoid == True:
            prob_mask = tf.sigmoid(self.slope * prob_mask)
        
        return prob_mask
    
    def write_probMask(self, mask, de_apply_sigmoid = True):
        
        """
        Returns the prob_mask
        Args:
            mask: np.array, the maks to write
            de_apply_sigmoid: Bool, whether the input "mask" is the prob_mask
                (in [0,1]) or the parameters that generate the prob_mask 
                after the sigmoid is applied.
        """
        
        if de_apply_sigmoid == True:
             mask = - tf.math.log((1. / mask) - 1.) / self.slope
        
        mask = tf.cast(tf.convert_to_tensor(mask), tf.float32)
        
        self.mult.assign(mask)
        
        return 
    
        
        
    def _logit_slope_random_uniform(self, shape, dtype=None, eps=0.01):
        # eps could be very small, or somethinkg like eps = 1e-6
        #   the idea is how far from the tails to have your initialization.
        # x = K.random_uniform(shape, dtype=dtype, minval=eps, maxval=1.0-eps) # [0, 1]
        x = tf.random.uniform(shape,minval = eps,maxval=1.0-eps)
        # logit with slope factor
        return - tf.math.log(1. / x - 1.) / self.slope
 
    
class ThresholdRandomMask(Layer):
    """ 
    Takes as input the prob_mask and a random generated vector, e.g., random
    uniform in [0,1]. It returns output = sigmoid(slope*(prob_mask-random)), 
    where "slope" is the hyper-parameter that controls the binary behaviour
    of the output mask.
    
    It compares the first input with the second element-wise, and for ech
    element that is greater returns a value that is close to 1, otherwise a
    value that is close to 0. How "close" depends on "slope".
    
    Notice that, the higher the slope the more binary the mask, but the more
    difficult the training (backprob is hampered).
    
    The output is the binary mask
    """

    def __init__(self, slope = 200, **kwargs):
        
        # The higher the "slope" the steeper the sigmoid, and a more
        # binary-like output mask. Too high "slope" hamper the backprop
        self.slope = slope
        
        super(ThresholdRandomMask, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'slope':self.slope})
        return config

    def build(self, input_shape):
        super(ThresholdRandomMask, self).build(input_shape)

    def call(self, x):
        inputs = x[0]
        thresh = x[1]
        
        return tf.sigmoid(self.slope * (inputs-thresh))


    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def class_name(self,):
        return 'ThresholdRandomMask'
    
class RandomMask(Layer):
    """ 
    Create a random mask of the same size as the input shape.
    The elements are randomly drown in the range [minval, maxval]
    """

    def __init__(self, minval = 0.0, maxval = 1.0, **kwargs):
        
        """
        Args:
            minval: Float, higerbound of the uniform distribution, default = 1
            minval: Float, lowebound of the uniform distribution, default = 0
            
        When the random behaviour requires to be halted, one can modify the two
        attributes by calling: "set_min_max(0.49999, 0.50001)".
        When one wants to give randomicity: "set_min_max(0.0, 1.0)".
        """
        
        self.minval = minval
        self.maxval = maxval
        super(RandomMask, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'minval':self.minval,
            'maxval':self.maxval})
        return config

    def build(self, input_shape):
        super(RandomMask, self).build(input_shape)

    def call(self,x):
        input_shape = tf.shape(x)
        threshs = K.random_uniform(
            input_shape,
            minval=self.minval,
            maxval=self.maxval,
            dtype='float32',
        )
        
        return threshs

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def set_min_max(self, maxval = None, minval = None):
        if maxval != None:
            self.maxval = maxval
        if minval != None:
            self.minval = minval
        return



class Multiply_with_mask(Layer):
    """
    Under-sampling by element-wise multiplication: it takes
    as input a tensor and a mask.
    
    If the mask is binary then it performs a selecetion of the input samples.
    
    If the mask is not binary, the layer can turn it binary by setting
    "hardThreshold = True".
    
    """

    def __init__(
            self,
            hardThreshold = False,
            mirror = False, 
            activate_pruning = True, # if False, the layer returns the first input
            threshold_value = 0.5,
            **kwargs,
        ):
        
        """
        Args:
            hardThreshold: Bool, whether to turn the mask binary or not
            mirror: Bool, whether to invert the values of the mask from 1-->0
                and 0-->1.
            activate_pruning: Bool, whether to multiply the input with the 
                mask or to ignore the mask and return the unchanged input.
            threhsold_value: Float, the value used to binarize tha mask, 
                only used when "hardThreshold = True".
        """
        
        self.hardThreshold = hardThreshold
        self.mirror = mirror
        self.activate_pruning = activate_pruning
        self.threshold_value = threshold_value
        super(Multiply_with_mask, self).__init__(**kwargs)
    
    def get_config(self):
        config = super(Multiply_with_mask,self).get_config().copy()
        config.update({'hardThreshold':self.hardThreshold,
                       'mirror':self.mirror,
                       'activate_pruning':self.activate_pruning,
                       'threshold_value':self.threshold_value,
                      })
        return config

    def build(self, input_shape):
        super(Multiply_with_mask, self).build(input_shape)

    def call(self, inputs):
        
        # to use flags, turn them into tensors.
        mirror = tf.cast(self.mirror, tf.float32)
        hardThreshold = tf.cast(self.hardThreshold, tf.float32)

        # whether to switch the 1s with the 0s
        mask = (1-mirror) * inputs[1] + mirror * (1-inputs[1])  # mirror applies NOT(mask) --- set False
        
        # Whether to binarize the mask
        mask = (1-hardThreshold) * mask + hardThreshold * tf.cast(tf.keras.backend.greater(mask, self.threshold_value),tf.float32)

        outputs = tf.multiply(inputs[0], mask)
        
        
        pruning = tf.cast(self.activate_pruning, tf.float32)
        outputs = inputs[0] * (1-pruning) + outputs * pruning
        
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def set_activate_pruning(self, activate_pruning):
        self.activate_pruning = activate_pruning
        return
    
    def set_hardThreshold(self, hardThreshold):
        self.hardThreshold = hardThreshold
        return
    
    def set_threshold_value(self, threshold_value = 0.5):
        self.threshold_value = threshold_value
        return

    
class Dropout_for_pruning(tf.keras.layers.Dropout):
    
    """
    A normal Dropout layer that can be de-activated by changing the 
    flag "activate".
    """
    
    def __init__(
            self,
            rate,
            activate = True,
            noise_shape=None,
            seed=None, 
            **kwargs,
        ):
        super(Dropout_for_pruning, self).__init__(
            rate, noise_shape, seed, **kwargs)
        
        self.activate = activate
        
        return
    
    def get_config(self):
        
        config = super().get_config().copy()
        
        config.update({'activate':self.activate})
        
        return config
    
    def build(self, input_shape):
        super(Dropout_for_pruning, self).build(input_shape)
    
    def call(self, inputs, training=None):  
        if self.activate == True:
            output = super(Dropout_for_pruning, self).call(inputs, training)
        else:
            output = inputs
            
        return output
        
    def set_activation(self, activate_dropout):
        
        self.activate = activate_dropout
        return
    
    
class Identity(Layer):
    """
    Returns the input unchanged.
    
    Used as a handy function to define the input of tf.keras.Model.
    """
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        return config

    def build(self, input_shape):
        super(Identity, self).build(input_shape)

    def call(self, x):
        return  x

    def compute_output_shape(self, input_shape):
        return input_shape