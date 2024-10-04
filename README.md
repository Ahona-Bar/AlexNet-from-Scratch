# Project Synopsis 
Using PyTorch, this project builds the AlexNet architecture from the ground up. One of the first deep convolutional neural networks, AlexNet, won the 2012 ImageNet competition with amazing results. Here, the FashionMNIST dataset is used to train AlexNet to categorize fashion items into ten distinct groups. 
# Tech Stack: 
## Framework: Pytorch
## Language: Python
## Libraries 
### torch: For constructing the model for deep learning. 
### torchvision: To load and preprocess the FashionMNIST dataset.
### tqdm: To show the status of the training. 
### torchsummary: For condensing the architecture of AlexNet. 
### nbimporter: Modules can be imported into Jupyter notebooks using this.
# Dataset
## Dataset Used: FashionMNIST
## Number of Classes: 10 (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)
## Number of Images:
## Training set: 60,000 images.
## Test set: 10,000 images.
## Image Size: The images are resized to 227x227 before being fed into the AlexNet model.
# Model Building
The layers that comprise the AlexNet architecture are as follows:
## Convolutional Layers:
### Conv1: 96 filters, 11x11 kernel, stride 4, padding 2.
### Conv2: 256 filters, 5x5 kernel, stride 1, padding 2.
### Conv3: 384 filters, 3x3 kernel, stride 1, padding 1.
### Conv4: 384 filters, 3x3 kernel, stride 1, padding 1.
### Conv5: 256 filters, 3x3 kernel, stride 1, padding 1.
## Pooling and Normalization:
### MaxPooling is applied after Conv1, Conv2, and Conv5.
### Local Response Normalization (LRN) is applied after Conv1 and Conv2.
## Fully Connected Layers:
### FC1: 4096 neurons.
### FC2: 4096 neurons.
### FC3: 100 neurons.
## Activation and Dropout:
### Following each convolutional and fully connected layer comes ReLU activation.
### To avoid overfitting, dropout is applied to the completely linked layers.
# Future Improvements
## Experiment with ImageNet:The ImageNet dataset served as the training set for the original AlexNet. While FashionMNIST training makes experimenting simpler, ImageNet can yield more insightful outcomes.
## Data augmentation: Adding random crops, flips, and rotations to the data could enhance the performance of the model even further.
## Reduce Learning Rate: Especially in longer training sessions, adding a more severe learning rate decay could improve convergence.
#References
## Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NIPS, 2012.
## PyTorch Documentation
## FashionMNIST Dataset
