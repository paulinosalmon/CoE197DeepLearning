# CoE197Z-SRU Deep Learning (2015-11557)
## Models
- 3-layer Multilayer Perceptron
- 3-layer Convolutional Neural Network
## Dataset
- CIFAR-10 Dataset

## Comparison
#### I. MLP
- Upon training and adjusting modifications for optimal parameters/hyperparameters of the MLP model for the CIFAR-10 dataset, results show that this model can achieve an accuracy of **56.44%**. The MLP model takes flattened vectors as inputs, and is inefficient in modern day advanced computer vision tasks such as image classification (CIFAR-10) as it has a characteristic that all its neurons are *fully connected*, leading to the case where its total number of parameters can grow very high, affecting results.

#### II. CNN
- Upon modifying the CNN model for the CIFAR-10 dataset, results show that this model can achieve an accuracy of **81.58%**, a lot higher as compared to the MLP model. In the context of image recognition, a CNN is more suited for this task as it takes the dimensonal information of a picture into account, as opposed to the MLP. Additionally, as its name goes, CNNs have a feature wherein, it downsamples the image first by convolution then uses a prediction layer at the end, drastically improving results.

## Future Improvements
- Data Augmentation
- Learning Rate Scheduling