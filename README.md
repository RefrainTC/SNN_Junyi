
# SNN_Junyi
For the collaboration of Junyi's SNN project.

Code includes computing thresholds for latency T=1 and calibrating the converted SNN model.

All the models are trained using PyTorch Framework.



One can use already trained ANN model for SNN conversion or can train ANN models using train_ann.py  file.

---------------------------------------------------------------------------------------------------------
python3 train_ann.py --dataset cifar10/cifar100 --model vgg16/resnet18/resnet20/cifarnet 


---------------------------------------------------------------------------------------------------------
To compute the threshold and initial potential values, 

python3 feature_extraction.py ----iter 1 --samples 100 --model vgg16/resnet18/resnet20/cifarnet --dataset  cifar10/cifar100 
--checkpoint dir-name

# iter - Number of iterations required to find the membrane potential and initial potential values.
# samples- Number of training data points used for computing the optimal threshold and initial potential values. 
dir-name - path to the directory where the trained models are stored.

In feature_extraction.py , to accelerate the computation speed we used a C routine and can be found in test.c file. test.so file
is also included, if it shows an error while loading the .so file, please compile the test.c file to generate .so file.
---------------------------------------------------------------------------------------------------------
For calibrating the converted SNN model, 
python3 train_snn_converted.py --model vgg16 --dataset cifar10 --t 1 --epochs 50.

For t=1, it trains an SNN model intialized with the weights of ANN model.
For t>1, it trains an SNN model intialized with the weights of SNN model with latency t-1. 
For t>1, 20 epochs are enough.

=======
