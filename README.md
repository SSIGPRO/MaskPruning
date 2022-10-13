# TurboMaskPruning

## General Description
We propose a novel pruning technique based on trainable probability masks that, when binarized, select the elements of the network to prune.
Our method features _i_) an automatic selections of the elements to prune by jointly training the binary masks with the model, _ii_) the capability of controlling the pruning level through hyper-parameters of a novel regularization term.

We have succesfully applied our method to prune the dense layers of AlexNet: check the Jupyter Lab demo!

This is a first version of a project that will be integrated and expanded in future, i.e., we plan to make our pruning masks work for many more layer types.

## Prunining by Masking
We make use of trainable binary masks in our work, that we first encounterd in [1] ,[2].

Binary masks are able to create binary structures by searching for the probability of every element of the final mask to be 0 or 1. Using probability masks allows to _i_) create non-binary and differentiable masks at training time, so the model can be trained (backpropagation would be hampered if binary object were used), and _ii_) to create binary masks at inference time, so to guarantee real pruning.

We train the resulting model by adding a novel regularization term in the loss, whose purpose is to push to binary values the elements of probability mask.

## Motivation of our method
In this work each dense layer of a neural network is coupled with a binary mask that learns what elements should be kept (1s in the mask) and what should be discarded (0s in the mask). The great and real advantage of our method is that model weights and mask weights are trained at the same time, so to run a pruning strategy that literally _i_) prunes only the weights that would make the loss decrease, and _ii_) re-adapts the non-pruned weights, in order to decrese the loss.

## Mask Pruning library
A new class "Model_for_pruning" is created. It inherits from "tf.keras.Model" and adds many functionalities:
1) given a model, it authomatically couples all the Dense layers with a trainable binary mask layer;
2) lets the user easily modify the internal parameters of the network, also proving shortcuts to methods used for visualization;
3) when the training is over, authomatically returns the pruned model by removing the masks and by only keeping the layers/weights corresponding to the "1s" of the masks.

The result is a handy collection of functions, all encompassed in the class "Model_for_pruning". Check the demo for a quick demonstration based on AlexNet.

## References
[1] https://github.com/cagladbahadir/LOUPE

[2] C. D. Bahadir, A. Q. Wang, A. V. Dalca and M. R. Sabuncu, "Deep-Learning-Based Optimization of the Under-Sampling Pattern in MRI," in IEEE Transactions on Computational Imaging, vol. 6, pp. 1139-1152, 2020, doi: 10.1109/TCI.2020.3006727.
