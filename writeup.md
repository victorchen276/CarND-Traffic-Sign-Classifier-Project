# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


[output1]: ./output/output_1.png "ouput_1"
[output2]: ./output/output_2.png "ouput_2"
[output3]: ./output/output_3.png "ouput_3"
[output4]: ./output/output_4.png "ouput_4"
[output5]: ./output/output_5.png "ouput_5"
[output6]: ./output/output_6.png "ouput_6"

[test1]: ./test_images/test_1.png "test1"
[test2]: ./test_images/test_2.png "test2"
[test3]: ./test_images/test_3.png "test3"
[test4]: ./test_images/test_4.png "test4"
[test5]: ./test_images/test_5.png "test5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.  
It shows a random set of images with the corrsponding names from csv file. (Cell 4 of the IPython notebook)



![alt text][output1]

The following cell (Cell 5) plot the occurrence of each image class. It shows how the data is distributed.
his can help understand where potential pitfalls could occur if the dataset isn't uniform in terms of a baseline occurrence.

![alt text][output2]
![alt text][output3]
![alt text][output4]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


1: Converting to grayscale - This worked well for Sermanet and LeCun as described in their traffic sign 
classification article. It also helps to reduce training time

2: Normalizing the data to the range (-1,1) - This was done using the line of code X_train_normalized = (X_train - 128)/128. 

here is the result

![alt text][output5]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution           | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Convolution           | 1x1 stride, same padding, outputs 1x1x400     |
| RELU					|												|
| Flatten               | Input = 5x5x16. Output = 400.                 |
| Dropout	            | 50% keep                                      |
| Fully connected		| Input = 800. Output = 43          			|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I  I used the AdamOptimizer with a learning rate of 0.0009. The epochs is 60 and the batch size was 100.

![alt text][output6]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.943
* test set accuracy of 0.936

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    I used a very similar architecture to the paper offered by the instructors. 
    
* What were some problems with the initial architecture?
    The first issue was lack of data for some images and the last was lack of knowledge of all the parameters. After I fixed those issues the LeNet model given worked pretty well with the defaults. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    Past what was said in the previous question, I didn't alter much past adding a couple dropouts with a 50% probability.
    
* Which parameters were tuned? How were they adjusted and why?
Epoch, learning rate, batch size, and drop out probability were all parameters tuned along with the number of random modifications 
to generate more image data was tuned. For Epoch the main reason I tuned this was after I started to get better accuracy early 
on I lowered the number once I had confidence I could reach my accuracy goals. The batch size I increased only slightly since starting 
once I increased the dataset size. The learning rate I think could of been left at .001 which is as I am told a normal starting point, 
but I just wanted to try something different so .00097 was used. I think it mattered little. The dropout probability mattered a lot early on, 
but after awhile I set it to 50% and just left it. The biggest thing that effected my accuracy was the data images generated 
with random modifications. This would turn my accuracy from 1-10 epochs from 40% to 60% max to 70% to 90% within the first 
few evaluations. Increasing the dataset in the correct places really improved the max accuracy as well.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

 I think I could go over this project for another week and keep on learning. I think this is a good question and I could still learn more about that. I think the most important thing I learned was having a more uniform dataset along with enough convolutions to capture features will greatly improve speed of training and accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test1] ![alt text][test2] ![alt text][test3] 
![alt text][test4] ![alt text][test5]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][test1] 

    40: Roundabout mandatory                                (95.38%)
    28: Children crossing                                   ( 4.61%)
    12: Priority road                                       ( 0.02%)
    7: Speed limit (100km/h)                                ( 0.00%)
    42: End of no passing by vehicles over 3.5 metric tons  ( 0.00%)
  
 
![alt text][test2] 

    1: Speed limit (30km/h)      (100.00%)
    0: Speed limit (20km/h)      ( 0.00%)
    2: Speed limit (50km/h)      ( 0.00%)
    3: Speed limit (60km/h)      ( 0.00%)
    4: Speed limit (70km/h)      ( 0.00%)

![alt text][test3] 
 
    12: Priority road            (100.00%) 
    9: No passing                ( 0.00%) 
    40: Roundabout mandatory     ( 0.00%) 
    0: Speed limit (20km/h)      ( 0.00%) 
    1: Speed limit (30km/h)      ( 0.00%) 

![alt text][test4] 

    17: No entry                 (100.00%)
    14: Stop                     ( 0.00%)
    0: Speed limit (20km/h)      ( 0.00%)
    1: Speed limit (30km/h)      ( 0.00%)
    2: Speed limit (50km/h)      ( 0.00%)

![alt text][test5] 

    38: Keep right               (100.00%)
    0: Speed limit (20km/h)      ( 0.00%)
    1: Speed limit (30km/h)      ( 0.00%)
    2: Speed limit (50km/h)      ( 0.00%)
    3: Speed limit (60km/h)      ( 0.00%)

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

answered in the privous one.


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


