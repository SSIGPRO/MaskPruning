#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:50:13 2020

@author: filippomartinini
"""

# third party
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Input

from tensorflow.keras.layers import Layer, Activation, LeakyReLU
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Multiply
from tensorflow.keras.layers import BatchNormalization, Concatenate, Add
from tensorflow.keras.layers import Subtract, Dense, Flatten

from tensorflow.keras.initializers import Initializer
from modules_demo import layers

import copy


class Model_pruning(tf.keras.Model):
    
    def verbose_layer_name(self, l, verbose):
        
        """
        Prints the layer name is verbose is True. Only used inside other
        functions.
        Args:
            l: Layer class
            verbose: Bool, whether to print the name of the layer class or not
        """
        
        if verbose == True:
            print(l.name)
        return l
    
    def activate_pruning(
            self,
            verbose = False,
        ):
        
        """
        Activates the mask-related layers
        Args:
            verbose: Bool, whether to print the name of the layers
                that are modified.
        """
        
        if verbose == True:
            print('\nActivate pruning:')
        
        for sub_model in self.layers:
            
            if 'pruning_mask' in sub_model.name:

                [self.verbose_layer_name(l, verbose).set_activate_pruning(True)  
                 for l 
                 in sub_model.layers 
                 if 'multiply_with_mask' in l.name
                ]
        
        return
    
    def deactivate_pruning(
            self,
            verbose = False,
        ):
        
        """
        Turns the mask-related layers to mere identity layers
        Args:
            verbose: Bool, whether to print the name of the layers
                that are modified.
                
        The layer "multiply_with_mask" ignores the input mask and simply
        returns the other input without modifying it.
        """
        
        if verbose == True:
            print('\nDeactivate pruning:')
        
        for sub_model in self.layers:
            
            if 'pruning_mask' in sub_model.name:

                [self.verbose_layer_name(l, verbose).set_activate_pruning(False)  
                 for l 
                 in sub_model.layers 
                 if 'multiply_with_mask' in l.name
                ]
        return
        
    def set_model_for_training(
            self,
            verbose = False,
        ):
        
        """
        Arrange the model to train, by randomizing the masks-related layers
        outputs, also assuirng masks are not binary.
        Args:
            verbose: Bool, whether to print the name of the layers
                that are modified.
                
        The mask behaviour is set random by setting the generation of random
        elements of layer "RandomMask" to the range [0.0, 1.0].
        """
        
        if verbose == True:
            print('\nSet hard threshold - set_min_max:')
        
        for sub_model in self.layers:
            
            if 'pruning_mask' in sub_model.name:
                
                [self.verbose_layer_name(l, verbose).set_hardThreshold(hardThreshold=False)
                 for l 
                 in sub_model.layers 
                 if 'multiply_with_mask' in l.name
                ]
                
                [self.verbose_layer_name(l, verbose).set_min_max(minval=0.0, maxval=1.0)  
                 for l 
                 in sub_model.layers 
                 if 'random_mask' in l.name
                ]

        return

    def set_model_for_inference(
            self,
            threshold_value = 0.5,
            verbose = False,
            minval = 0.4999,
            maxval = 0.5001,
        ):
        
        """
        Arrange the model to infere, by stabilizing the masks-related layers
        outputs (that are otherwise stochastic), also binarizing the masks.
        Args:
            threshold_value: Float in [0,1], the threshold used to binarize the masks
                mask[i] = 1, if mask[i] >= threshold; 0, otherwise; for all i
            verbose: Bool, whether to print the name of the layers
                that are modified.
                
        The mask behaviour is stablized by setting the generation of random
        elements of layer "RandomMask" to the range [0.4999, 0.5001].
        """
        
        if verbose == True:
            print('\nSet_hard_threshold - set_threshold_value - set_min_max:')
        
        for sub_model in self.layers:
            
            if 'pruning_mask' in sub_model.name:
        
                # activate Hard Threshold
        
                [self.verbose_layer_name(l, verbose).set_hardThreshold(hardThreshold=True)  
                 for l
                 in sub_model.layers 
                 if 'multiply_with_mask' in l.name
                ]

                [self.verbose_layer_name(l, verbose).set_threshold_value(threshold_value)  
                 for l
                 in sub_model.layers 
                 if 'multiply_with_mask' in l.name
                ]

                # activate mask randomicity

                [self.verbose_layer_name(l, verbose).set_min_max(minval=minval, maxval=maxval)  
                 for l
                 in sub_model.layers 
                 if 'random_mask' in l.name
                ]
        
        return
    
    def set_mask_trainability(
            self,
            mask_trainability,
            verbose = False,
        ):
        
        """
        Set the trainability of all the mask-related layers
        Args:
            mask_trainability: Bool, whether to train or not
            verbose: Bool, whether to print the name of the layers
                that are modified.
        """
        
        if verbose == True:
            print('\nmask_trainability:')
        
        for sub_model in self.layers:
            
            if 'pruning_mask' in sub_model.name:
        
                l_list = [
                        self.verbose_layer_name(l, verbose)
                        for l
                        in sub_model.layers 
                        if 'prob_mask' in l.name
                    ]

                for l in l_list:
                    l.trainable = mask_trainability
        
        return

    
    def set_normal_layers_trainability(
            self,
            layers_trainability,
            verbose = False,
        ):
        
        """
        Set the trainability of all the non-mask-related layers
        Args:
            layers_trainability: Bool, whether to train or not
            verbose: Bool, whether to print the name of the layers
                that are modified.
        """
        
        if verbose == True:
            print('\nNormal layers trainability:')
        
        for sub_model in self.layers:
            
            if 'sub_model' in sub_model.name:
                
                l_list = [
                        self.verbose_layer_name(l, verbose)
                        for l
                        in sub_model.layers 
                        if not('prob_mask' in l.name)
                    ]

                for l in l_list:
                    l.trainable = layers_trainability
        
        return
                
    
    def read_prob_masks(
            self,
            apply_sigmoid = True,
            verbose = False,
        ):
        
        """
        Reads all the probability masks.
        Args:
            apply_sigmoid: Bool, whether the prob_masks returned in 
                their probability [0,1] shape or in their
                real unconstrained shape (in R).
            verbose: Bool, whether to print the name of the layers
                that are modified.
        """
        
        if verbose == True:
            print('\nread prob masks:')
        
        prob_masks_list = []
        
        for sub_model in self.layers:
            
            if 'pruning_mask' in sub_model.name:
                
                prob_masks = [
                        self.verbose_layer_name(l, verbose).read_probMask(apply_sigmoid)
                        for l
                        in sub_model.layers
                        if 'prob_mask' in l.name
                    ]
            
                prob_masks_list += prob_masks
        
        return prob_masks_list
    
    
    def write_prob_masks(
            self,
            prob_masks,
            de_apply_sigmoid = True,
            verbose = False,
        ):
        
        """
        Rewrites all the probability masks.
        Args:
            prob_masks: list of float, the prob_masks to write.
            de_apply_sigmoid: Bool, whether the prob_masks are probabilities
                in [0, 1] or real unconstrained values in R.
            verbose: Bool, whether to print the name of the layers
                that are modified.
        """
        
        if verbose == True:
            print('\nWrite prob masks:')
        
        l_list = []
        
        for sub_model in self.layers:
            
            if 'pruning_mask' in sub_model.name:
                
                l_tmp = [
                    self.verbose_layer_name(l, verbose)
                    for l
                    in sub_model.layers
                    if 'prob_mask' in l.name
                ]
                
                l_list += l_tmp
        
        assert len(l_list) == len(prob_masks), 'len(actual_prob_masks) != len(prob_masks_to_write)'
        
        for l, m in zip(l_list, prob_masks):
            l.write_probMask(m, de_apply_sigmoid)
        
        return
    
    def read_inference_masks(
            self,
            threshold_value = None,
            verbose = False,
            minval = 0.4999,
            maxval = 0.5001,
            silece_alert = False,
        ):
        
        """
        Returns the weights of all the dense layers.
        Args:
            threshold_value: float in [0,1], threshold used
            to binarize the masks.
            verbose: Bool, whether to print the name of the layers
                that are modified.
            silece_alert: Bool, whether to silence the warning of 
                "model shifter to inference"
            
        """
        
        
        if verbose == True:
            print('\nInference Masks:')
        
        inputs = self.inputs
        
        self.set_model_for_inference(
                threshold_value = threshold_value,
                minval = minval,
                maxval = maxval,
            )
        if silece_alert == False:
            print('!!!!\t"',self.name,'" has been shifted to "INFERENCE" mode.')
        
        outputs_list = []
        
        for sub_model in self.layers:
            
            if 'pruning_mask' in sub_model.name:
                
                outputs = [
                        sub_model.get_layer(self.verbose_layer_name(l, verbose).name).output
                        for l
                        in sub_model.layers
                        if 'sampled_mask' in l.name
                    ]
            
                outputs_list += outputs
        
        fake_input = np.zeros((1, )+tuple(self.input.shape)[1:])
        
        masks = tf.keras.Model(
                inputs,
                outputs_list,
                name = 'model_masks',
            ).predict(fake_input)
        
        if threshold_value is not None:
            masks = np.array(masks) > threshold_value
        
        return masks
    
    
    def get_dense_weights(
            self,
        ):
        
        """
        Returns the weights of all the dense layers.
        Args:
            
        """
        
        dense_weights_list = []
        
        for sub_model in self.layers:
        
            if 'sub_model' in sub_model.name:

                dense_weights = [
                        l.get_weights()
                        for l
                        in sub_model.layers
                        if 'Dense' in str(l)
                    ]

                dense_weights_list += dense_weights
        
        return dense_weights_list
    
    
    def return_pruned_model(
            self,
            threshold_to_binarize_mask_values = 0.5,
            minval = 0.49999,
            maxval = 0.50001,
            model_name = 'pruned_model',
        ):
        
        """
        Returns the version of the initial model with pruned dense layers.
        Args:
            threshold_to_binarize_mask_values: float in [0,1], threshold used
            to binarize the masks
            model_name: String, name of the model to return.
        This function removes all the mask-related layers and redefines the 
        dense layers with the non-pruned number of neurons, only mantaining
        the non-pruned weights.
        """
        
        #### PREPARE TO CREATE THE PRUNED MODEL
        
        # get all the weights of the Dense layers
        dense_weights_list = self.get_dense_weights()
        
        # get the binarized masks
        inference_masks = self.read_inference_masks(
                threshold_value = threshold_to_binarize_mask_values,
                minval = minval,
                maxval = maxval,
            )
        
        # returnes the non-pruned weights of the dense layers
        weights_pruned = []
        for i, d  in enumerate(dense_weights_list):
            kernel_pruned = np.array(d[0])
            bias_pruned = np.array(d[1])

            # the "if conditions" handle the first and last dense pruning
            if i<len(dense_weights_list)-1:
                kernel_pruned = kernel_pruned.T[inference_masks[i][0]].T
                bias_pruned = bias_pruned[inference_masks[i][0]]
            if i>0:
                kernel_pruned = kernel_pruned[inference_masks[i-1][0]]

            weights_pruned += [[kernel_pruned, bias_pruned]]
        
        #### CREATE THE PRUNED MODEL
        
        inputs = self.inputs
        last_tensor = inputs[0]
        counter = -1

        # name of the last "sub_model"
        last_sub_model_name = [
                m
                for m
                in self.layers
                if 'sub_model' in m.name
            ][-1].name

        # this cicle avoids all the sub_models containing mask-related layers
        for sub_model in self.layers:

            if 'sub_model' in sub_model.name:

                for l in sub_model.layers[1:-1]: 

                    last_tensor = l(last_tensor)

                counter += 1

                # the last layer of a "sub_model" is a dense
                old_dense = sub_model.layers[-1]

                # number of non-pruned neurons to set as units for the dense
                if sub_model.name != last_sub_model_name:
                    new_units = np.sum(inference_masks[counter])

                # last dense layer does not have an associated mask
                else:
                    new_units = old_dense.units

                dense = tf.keras.layers.Dense(
                        units = new_units,
                        activation = old_dense.activation,
                        use_bias = old_dense.use_bias,
                        kernel_initializer = old_dense.kernel_initializer,
                        bias_initializer = old_dense.bias_initializer,
                        kernel_regularizer = old_dense.kernel_regularizer,
                        bias_regularizer = old_dense.bias_regularizer,
                        activity_regularizer = old_dense.activity_regularizer,
                        kernel_constraint = old_dense.kernel_constraint,
                        bias_constraint = old_dense.bias_constraint,
                        name = old_dense.name,
                    )

                last_tensor = dense(last_tensor)

        model_pruned = tf.keras.Model(
                inputs = inputs, 
                outputs = last_tensor, 
                name = model_name,
            )
        
        # SET THE WEIGHTS OF THE DENSE LAYERS
        # Only the non-pruned weights are tranferred
        
        counter = -1
        for l in model_pruned.layers:
            if 'Dense' in str(l):
                counter += 1
                l.set_weights(weights_pruned[counter])
        
        return model_pruned
        
    
def model_AlexNet(
            input_shape,
            units_output = 100,
            trainable_neurons = True,
            use_data_augmentation_layers = True,
            name = 'AlexNet',
        ):
    
    """
    Returns a model of AlexNet. 
    Args:
        inputs_shape: Tuple, the shape of one input; note that (224, 224, 3)
            is the image shape for which AlexNet has been designed.
        units_output: Int, the number of the classes, i.e., the number
            of neurons of the last dense layers (to do classisification).
        trainable_neurons: Bool, whether to set the dense trainable or not.
        use_data_augmentation_layers: Bool, whether to add some tf data
            augmentation layers.
        name: String, the name of the model
    
    Returms:
        a tf.keras.Model
    """
    
    input_tensor = Input(
            shape=input_shape, 
            name="input",
        )
        
    if use_data_augmentation_layers == True:
                
        last_tensor = tf.keras.layers.RandomContrast(
                factor = (0.3, 0.7),
                name = 'random_contrast',
            )(input_tensor)
        last_tensor = tf.keras.layers.RandomFlip(
                mode = 'horizontal',
                name = 'random_flip',
            )(last_tensor)
        last_tensor = tf.keras.layers.RandomRotation(
                factor = 0.1, 
                name = 'random_rotation',
            )(last_tensor)
        last_tensor = tf.keras.layers.RandomTranslation(
                height_factor = 0.1,
                width_factor = 0.1,
                fill_mode = 'nearest',
            )(last_tensor)
        
    else:
        last_tensor = copy.deepcopy(input_tensor)
    
        
    ### CONVOLUTIONAL LAYERS
    
    filters_list = (96, 256, 384, 384, 256)
    kernel_size_list = ((11, 11), (5, 5), (3, 3), (3, 3), (3, 3))
    strides_list = (4, 1, 1, 1, 1)
    pool_list = (True, True, False, False, True)
    
    iterate = zip(
            filters_list,
            kernel_size_list,
            strides_list,
            pool_list,
        )
    
    for id_tmp, (filters, kernel_size, strides, pool) in enumerate(iterate):
        last_tensor = Conv2D(
                filters = filters,
                kernel_size = kernel_size,
                strides = strides,
                padding = 'same',
                activation = 'relu',
                name = 'conv_'+str(id_tmp),
            )(last_tensor)

        last_tensor = tf.keras.layers.BatchNormalization(
                name = 'batch_norm_'+str(id_tmp),
            )(last_tensor)

        if pool == True:
            last_tensor = MaxPooling2D(
                    pool_size=(3, 3),
                    strides = 2,
                    name = 'pool_'+str(id_tmp),
                )(last_tensor)
    
    
    last_tensor = Flatten(
            name = 'flatten',
        )(last_tensor)
    
    ### DENSE LAYERS
        
    for id_tmp in range(2):
        
        id_tmp = id_tmp + len(pool_list)
        
        last_tensor = Dense(
                units = 4096,
                activation="relu", 
                name='dense_'+str(id_tmp),
            )(last_tensor)
            
        last_tensor = tf.keras.layers.Dropout(
                rate = 0.5,
                name = 'dropout_'+str(id_tmp)
            )(last_tensor)
        
    outputs = Dense(
            units_output, 
            activation="softmax",
            name = 'prediction_output'
        )(last_tensor)

   
    alexnet = tf.keras.Model(
            inputs = input_tensor, 
            outputs = outputs,
            name = name,
        )

    return alexnet


def pruning_trainable_mask(
        input_tensor,
        slope = 200, 
        hardThreshold= False,
        identifier = '', 
        trainable = True,
        prob_mask_list = [],
        activate_pruning = True,
    ):
    
    """
    Takes a tf.Tensor as input, implements all the layers that:
        - create a mask having the same shape of the input
        - multiplies the input with the mask
    and gathers all the layers into a tf.keras.Model.
    It returns the output of the model, so that it can be used to 
    create a bigger model.
    
    Args:
        input_tensor: Tuple, the tensor to multiply with the mask
        slope: Float, the multiplicative factor used to binarize the mask at
            the end of the mask generation process (the higher the more 
            binarysh the mask, but more difficult the training); note that 
            slope = 200 is a standard value (result of tuning).
        hardThreshold: Bool, whether the masks are binary or almost binary
            (binary mask can be used only during inference)
        identifier: Int, a value added to the name of the layers
            to distinguish them between other identical layers.
        trainable: Bool, whether the prob_mask is trainable of frozen.
        prob_mask_list: List, used to authomatically collect all the prob_mask
            produced by a model when this function is called multiple times;
            useful to give the prob_mask_list as input to the function 
            "pruning_model_concatenate_masks" to complete the pruning model.
        activate_pruning: whether the model should activate pruning of not.
        
    """
    
    # identity layer does not nothing, but it is useful for the model creation
    identity_layer = layers.Identity(
            name = 'identity_'+str(identifier),
        )
    
    last_tensor = identity_layer(input_tensor)
    
    # probability mask
    prob_mask_layer = layers.ProbMask(
            name='prob_mask_'+str(identifier),       
            slope=5,      
            trainable=trainable,      
        )
    
    prob_mask_tensor = prob_mask_layer(last_tensor)

    # Realization of random uniform mask
    thresh_tensor = layers.RandomMask(
            name='random_mask_'+str(identifier),
        )(prob_mask_tensor) 

    # Realization of mask
    last_tensor_mask = layers.ThresholdRandomMask(
            slope=slope,
            name='sampled_mask_'+str(identifier),
        )([prob_mask_tensor, thresh_tensor])
    
    # multiplies element-wise the ipnut with the mask
    multiply_with_mask_layer = layers.Multiply_with_mask(
            name = 'multiply_with_mask_'+str(identifier),
            hardThreshold = hardThreshold,
            activate_pruning = activate_pruning,
        )
    
    last_tensor_pruned = multiply_with_mask_layer(
            [last_tensor, last_tensor_mask],
        )
    
    # Model
    model_pruning_mask = tf.keras.Model(
            inputs = identity_layer.input,
            outputs = [last_tensor_pruned, prob_mask_tensor],
            name = 'pruning_mask_'+str(identifier),
        )
        
    [last_tensor, prob_mask_tensor] = model_pruning_mask(input_tensor)
    
    prob_mask_list += [prob_mask_tensor]
    
    outputs = [last_tensor, prob_mask_list]
    
    return outputs


def pruning_model_concatenate_masks(
        prob_mask_list, 
    ):
    
    """
    Takes the list of all the probability masks used by a model
    and concatenates them together to produce an output that
    can be given as input to the regularization term in the loss function.
    The regularization term wants to:
        - reduce the number of ones in the masks (intensify the pruning)
        - binarize the masks (reduce their stochastic behaviour)
        
    Args:
        prob_mask_list: List, contains all the prob_masks used by the model
        
    Returns:
        the tensor ouput of the tf.keras.Model that concatenates the masks.
    """
    
    if len(prob_mask_list) >= 1:

        prob_mask_flat_list = []

        # by flattening the masks we can concatenate them
        flatten_layers_list = [
                tf.keras.layers.Flatten(
                        name = 'flatten_mask_'+str(i),
                    )
                for i 
                in range(len(prob_mask_list))
            ]
        
        prob_mask_flat_list = [
                flat(prob_mask)
                for flat, prob_mask
                in zip(flatten_layers_list, prob_mask_list)
            ]

        # concatenate the masks
        prob_mask_concatenated = tf.keras.layers.Concatenate(
                name = 'concatenate_pruning_masks',
            )(prob_mask_flat_list)
        
        model_concatenate_masks_inputs = [
                f.input
                for f
                in flatten_layers_list
            ]
        
        # Model
        model_concatenate_masks = tf.keras.Model(
                model_concatenate_masks_inputs,
                prob_mask_concatenated,
                name = 'model_concatenate_masks',
            )
        
        pruning_outputs = model_concatenate_masks(prob_mask_list)
    else:
        assert False, 'The model does not contain any Dense layer, pruning is only apllied to such layers.'
        
    return pruning_outputs  


def add_mask_pruning(
        model,
        model_name = 'model_pruning',
    ):
    
    """
    Takes a tf.keras.Model and returns a "Model_for_pruning" model, that 
    after every Dense layer applies pruning 
    by multiplying the Dense output with a trainable binary mask.
    (the last dense layer is not affected)
    
    Notice that this function only works with models built is a 
    sequential-like fashion, i.e., they must be like:
    a = input
    b = sub_model(a)
    b2 = dense(b)
    
    c = sub_model(b2)
    c2 = dense(c)
    
    ...
    ...
    
    z = sub_model(w2)
    output = dense(z)
    
    Sub_model can be as complex as possible, but their output/inputs 
    must be "low-complex enough". "sub_model"s should not be "Functional".
    
    Args:
        model: tf.keras.Model, the model one wants to prune.
        model_name: String, the name to give to the pruning model
        
    Returns:
        an instance of the class "Model_for_pruning" that couples every
        Dense with a mask so it can be pruned.
    """
    
    # name of the output layer
    name_output_layers = [
        str(m)[str(m).find('by layer ')+10:str(m).find('")')-1]
        for m 
        in model.outputs
    ]
    
    prob_mask_list = []
    
    inputs = model.inputs
    m_input = inputs
    last_tensor = model.inputs
    counter = 0
    
    for l in model.layers:
        
        # when the layer is Dense and is not the output
        if 'Dense' in str(l) and not(l.name in name_output_layers):
            
            counter += 1
            
            # everything between the input or mask (not included) 
            # and the dense (inlcuded) form a tf.keras.Model
            sub_model_tmp = tf.keras.Model(
                    m_input,
                    l.output,
                    name = 'sub_model_'+str(counter),
                )
            
            last_tensor = sub_model_tmp(last_tensor)
            
            # returns the mask and the updated prob_mask_list
            [last_tensor, prob_mask_list] = pruning_trainable_mask(
                    input_tensor = last_tensor,
                    slope = 200,
                    hardThreshold= False,
                    identifier = counter, 
                    trainable = True,
                    prob_mask_list = prob_mask_list,
                    activate_pruning = True,
                )
            
            # find the layer that 
            for l2 in model.layers:
                s = str(l2.input)
                if l.name in s:
                    m_input = l2.input
    
    # last sub_model
    sub_model_tmp = tf.keras.Model(
            m_input,
            model.outputs,
            name = 'sub_model_'+str(counter+1)
        )
    
    outputs = sub_model_tmp(last_tensor)
    
    # last part of the model that gathers the masks and produces an
    # output to be used as input to the regularization term in the loss.
    output_pruning = pruning_model_concatenate_masks(prob_mask_list)
    
    outputs = [outputs, output_pruning]
    
    # the pruning model
    model_pruning = Model_pruning(inputs, outputs, name = model_name)
    
    return model_pruning