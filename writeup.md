#**Traffic Sign Recognition** 
---
[image1]: ./Data%20Distribution.jpg "Data Distribution"
[image2]: ./Additional%20Images/1.jpg "Bumpy Road"
[image3]: ./Additional%20Images/2.jpg "No Entry"
[image4]: ./Additional%20Images/3.jpg "Yield"
[image5]: ./Additional%20Images/4.jpg "Stop"
[image6]: ./Additional%20Images/5.jpg "Slippery Road"

### Writeup

You're reading it! and here is a link to my [project code](https://github.com/akshatbjain/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb). Click [here](https://github.com/akshatbjain/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.html) if you prefer an exported html instead.

### Data Set Summary & Exploration

I used the pickle files and numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32, 1)
* The number of unique classes/labels in the data set is: 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![alt text][image1]

I also save a random image from the dataset as seen in the [ipython notebook](https://github.com/akshatbjain/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb).

### Design and Test a Model Architecture

#### Preprocessing:
1. As the first step, I decided to convert the images to grayscale because it reduces the dimensionality, thus increasing the training speed.

2. As the second step, I normalized the image data in order to provide numerical stability to the network. We never want the values involved in the calculation of our loss function to be too big or too small. Normalization provides conditioning to our input images. A badly conditioned problem means that the optimizer has to do a lot of searching to go and find a good solution. A well conditioned problem makes it a lot easier for the optimizer to do it's job. Normalization doesn't change the content of our image but it makes it much easier for the optimization to proceed numerically.


#### Model Architecture

My final model consisted of the following layers:

| Layer | Description | 
|:---------------------:|:---------------------------------------------:| 
| Input | 32x32x1 preprocessed image | 
| Convolution1 3x3 | stride: 1x1, padding: valid padding, activation: ReLU, output: 30x30x16 |
| Convolution2 3x3 | stride: 1x1, padding: valid padding, activation: ReLU, output: 28x28x16 |
| Convolution3 3x3 | stride: 1x1, padding: valid padding, activation: ReLU, output: 26x26x32 |
| Max Pooling | stride: 2x2, kernel: 2x2,  padding: valid padding, output: 13x13x32 |
| Dropout | 0.5 |
| Convolution4 3x3 | stride: 1x1, padding: valid padding, activation: ReLU, output: 11x11x32 |
| Convolution5 3x3 | stride: 1x1, padding: valid padding, activation: ReLU, output: 9x9x64 |
| Convolution6 3x3 | stride: 1x1, padding: valid padding, activation: ReLU, output: 7x7x64 |
| Max Pooling | stride: 2x2, kernel: 2x2,  padding: valid padding, output: 3x3x64 |
| Dropout | 0.5 |
| Flatten/Reshape | input: 3x3x64, output: 576 |
| Fully-connected1 | activation: ReLU, output: 256 |
| Dropout | 0.5 |
| Fully-connected2 | activation: ReLU, output: 128 |
| Dropout | 0.5 |
| Output | output: n_classes |

To train the model, I used the Adam Optimizer with a batch size of 32 for 15 epochs. The learning rate was initialized to 0.001. I used a dropout of 0.5 for training.

The model achieves an accuracy of 98+ in the first 10 epochs. The extra 5 epochs were added because in when I ran this experiment multiple times, it sometimes took 12 epochs to reach the 98+ mark.

My final model results were:
* training set accuracy of: 99.644%
* validation set accuracy of: 98.866%
* test set accuracy of: 96.952%

I first tried the LeNet model and found that it wasn't crossing the 93% accuracy mark. I then decided to tweak a bit with the LeNet architecture by playing around with maxpooling and dropout but found that, that wasn't helping either. I then decided to build my own model architecture and so I came up with the model shown above. The number of convolution and fully connected layers was fixed after a series of experiements and I found the above combination to give the best results.

I feel the problem with the LeNet architecture was that it didn't have enough convolutional layers to be able to extract enough information from the input images. Also, LeNet wasn't deep enough in order to be able to learn the numerous features present in the input images. My network solved both these problems and thus gave better results than LeNet.

I added max pooling and dropout after the third and sixth convolutional layers. Dropout was added after both the fully-connected layers. Max pooling helps reduce dimensionality and retains only the most important pieces of information. Dropout is an important regularization technique. With dropout the network can never rely on any given activation to be present as they can be squashed at any given time. The network is thus forced to learn a redundant representation for everything to make sure that at least some of the information remains. This makes the network robust and prevents overfitting. It also makes our network act as if it is taking consensus over an ensemble of neurons, which is always a good way to improve performance.

Also, I found that using a batch size of 32 made the model reach the 98+ mark faster than using a higher batch size. 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

I chose the above images as they were the easiest to find. I would definitely like to test the model on more images in future.

Here are the results of the prediction:

| Image	| Prediction	| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road	| Bumpy Road	| 
| No Entry	| No Entry	|
| Yield | Yield	|
| Stop	| Stop	|
| Slippery Road	| Slippery Road	|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of: 96.952%

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For the first image, the model is very sure that this is a bumpy road sign (probability of 0.999), and the image does contain a bumpy road sign. The top five soft max probabilities were

| Probability | Prediction | Sign Name |
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:| 
| .999 | 22 | Bumpy Road	| 
| .001	| 29	| Bicycles crossing |
| .000	| 25	| Road work |
| .000	| 20	| General caution |
| .000 | 34 | Turn left ahead |

For the second image, the model is absolutely sure that this is a no entry sign (probability of 1.0), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability | Prediction	|  Sign Name |
|:---------------------:|:---------------------------------------------:| :---------------------------------------------:| 
| 1.00 | 17 | No entry	| 
| .000	| 14	| Stop |
| .000	|  8	| Speed limit (120km/h) |
| .000	| 38	| Keep right |
| .000 | 12 | Priority road |

For the third image, the model is somewhat sure that this is a yield sign (probability of 0.426), and the image does contain a yield sign. The top five soft max probabilities were

| Probability | Prediction	|  Sign Name |
|:---------------------:|:---------------------------------------------:| :---------------------------------------------:| 
| .426 | 13 | Yield	| 
| .237	| 10	| No passing for vehicles over 3.5 metric tons |
| .061	|  2	| Speed limit (50km/h) |
| .060	| 12	| Priority road |
| .050 |  5 | Speed limit (80km/h) |

For the fourth image, the model is very sure that this is a stop sign (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability | Prediction	|  Sign Name |
|:---------------------:|:---------------------------------------------:| :---------------------------------------------:| 
| .999 | 14 | Stop	| 
| .001	|  3	| Speed limit (60km/h) |
| .001	| 33	| Turn right ahead |
| .000	|  5	| Speed limit (80km/h) |
| .000 | 34 | Turn left ahead |

For the fifth image, the model is very sure that this is a slippery road sign (probability of 0.932), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability | Prediction	|  Sign Name |
|:---------------------:|:---------------------------------------------:| :---------------------------------------------:| 
| .932 | 23 | Slippery road	| 
| .068	| 30	| Beware of ice/snow |
| .000	| 29	| Bicycles crossing |
| .000	| 20	| Dangerous curve to the right |
| .000 | 21 | Double curve |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
