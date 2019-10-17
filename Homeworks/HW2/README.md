# Homework 2: Image Restoration using Autoencoders

## Specifications
- **Input**: Input data is an MNIST image corrupted by a white square of at most 25% of the image area
- **Output**: Restored MNIST image w/o the corrupting white square.

## Dataset
- MNIST

## Discussion
### Training Parameters
- **Adam** has been chosen as the optimizer. Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems. It is also popular as it can achieve good results fast despite the vast number of features to be generated.
- **MSE** has been chosen as the loss function as it is commonly used for regression problems. This option also proved effective as it exceeded baseline expectations for the classifier and the SSIM scores.

### Model Design
- This model design is similar to that of the one from Sir Atienza's github repository. It has modular design, instantiating an encoder, a decoder and an autoencoder. This entire model takes in 28x28x1 images as inputs as shown in the Model Summary section of the notebook, and it outputs 28x28x1 images as well.

### Problems Encountered
- One of the problems I encountered was trying to create a corruption code for the white square. I couldn't figure out the matching dimensions for it nor I could fully understand how to manipulate the pixels. But this was solved after the code for applying noise was provided.
- Another problem I had was trying to run the benchmark code properly. For some reason, my classifier score spiked to 290% before. But it was fixed after I moved most of my code chunks to separate files, as I might have accidentally messed up the dimensions for my model.
