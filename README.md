# **Behavioral Cloning** 

## Writeup Report - Gaurav Garg

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The Model in this submmission is based on comma.ai' research https://github.com/commaai/research
The model consists of following layers
1. Normalizing Layer  - using lambda function as discussed in the classes 
2. Cropping Layer - Then the images were cropped to get the relevant data 
3. convolution neural network - with 8x8 filter sizes and depth of 16
4. ELU Layer
5. convolution neural network - with 5x5 filter sizes and depth of 32
6. ELU Layer
7. convolution neural network - with 5x5 filter sizes and depth of 64
8. Flatten Layer
9. Dropout Layer to avoid overfitting
10. ELU Layer
11. Fully Connected Layer
12. Dropout Layer
13. ELU Layer
14. Fully Connected Layer

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the leNet. I thought this model might be appropriate because it has worked well the classifier project. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I implemented aggressive dropout probabilities which helped a lot. Additionally lesser nuumber of epochs enabled me avoid overfitting as well

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Therefore it was required to do some data augmentation in order to improve the driving behavior. Data augmentation was done in the follwoing ways 
* For each data point , we random among the three camera positions (left, center, right) was chosen with a steering correction of 0.25. The value of steering correction is adjusted based on how training the network was behaving
* Images were also randomly flipped and also the changing of steering angle.

Only these 2 techniques were enough to augment the data and the car was finally driving autonomously


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes 

1. Normalizing Layer  - using lambda function as discussed in the classes 
2. Cropping Layer - Then the images were cropped to get the relevant data 
3. convolution neural network - with 8x8 filter sizes and depth of 16
4. ELU Layer
5. convolution neural network - with 5x5 filter sizes and depth of 32
6. ELU Layer
7. convolution neural network - with 5x5 filter sizes and depth of 64
8. Flatten Layer
9. Dropout Layer to avoid overfitting
10. ELU Layer
11. Fully Connected Layer
12. Dropout Layer
13. ELU Layer
14. Fully Connected Layer

#### 3. Creation of the Training Set & Training Process

I have used the sample driving data provided by udacity to train my network This consists of 8036 data samples . 85% of the samples were used to train while the remaining 15% was used for validation. Each data sample contains the steering measurement as well as three images captured from three cameras installed at three different locations in the car [left, center, right].

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by minimum change in loss after 10 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
