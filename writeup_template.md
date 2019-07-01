# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./writeup-images/images-per-class.png "# images per class"
[image2]: ./writeup-images/image-processing.png "Image processing"
[image3]: ./writeup-images/image-transforms.png "Randomizing a few factors to generate new validation data"
[image4]: ./writeup-images/sign-5.jpg "Traffic Sign 1"
[image5]: ./writeup-images/sign-11.jpg "Traffic Sign 2"
[image6]: ./writeup-images/sign-25.jpg "Traffic Sign 3"
[image7]: ./writeup-images/sign-32.jpg "Traffic Sign 4"
[image8]: ./writeup-images/sign-33.jpg "Traffic Sign 5"
[image9]: ./writeup-images/conv1-output.png "Output of the first convolutional layer"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used plain Python to calculate summary statistics of the traffic signs data set:

```python
n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = np.shape(X_train[0])
n_classes = len(np.unique(y_test))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I started with simply outputting the count of images per class, but ended up using a bar chart to visualize the density of test data for each class:

```python
def count_classes(labels):
    """
    Count the number of images in each class
    """
    data_breakdown = {}
    for i, img_class in enumerate(labels):
        if str(img_class) in data_breakdown.keys():
            data_breakdown[str(img_class)] = data_breakdown[str(img_class)] + 1
        else:
            data_breakdown[str(img_class)] = 1
    return data_breakdown

training_data_breakdown = count_classes(y_train)

num_images_per_class = []
for i in range(n_classes):
    num_images_per_class.append(training_data_breakdown[str(i)])

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)
ax.set_xticks(range(n_classes)) 
ax.bar(range(n_classes), num_images_per_class, align='center')
ax.set_title('Num images per class')
```

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

This was the gist of the image processing:
* Convert image to YUV color space
* Take the Y channel (discard the rest)
* Apply contrast limited adaptive histogram equalization (CLAHE) to this Y channel

```python
def preprocess(img):
    """
    Normalize brightness by converting to YUV and applying an aggressive CLAHE to the Y channel.
    The high clipLimit seems to be pretty aggressive with local contrast, which helps the signs stand out
    """
    clahe = cv2.createCLAHE(clipLimit=200.0, tileGridSize=(6,6))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    Y_channel = img[:,:,0]
    img[:,:,0] = clahe.apply(Y_channel)
    return np.array(img[:,:,0]).reshape(32,32,1)
```

![alt text][image2]

Finally, the CLAHE-processed image was normalized to both reduce the numeric variation in the pixel data as well as "center" it around 0 (values are -1.0 to 1.0 instead of 0 to 255)

```python
def normalize(img):
    return (img - 128) / 128
```

After some experimentation using the basic data set, I decided to create additional training data using
guidelines set forth in the [sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper.

To add more data to the training set, I applied a combination of:
* resizing the image in both the x and y axes by a random factor between 0.9 and 1.1,
* rotating the image a random amount between -15 and 15 degrees
* shifting the image between -2 and 2 pixels in both the x and y directions

```python

def jitter_position(img):
    """
    Randomly shift the image by up to 2 pixels in the x and/or y directions
    """
    x_d = random.randint(-2, 2)
    y_d = random.randint(-2, 2)
    rows, cols, channels = img.shape
    M = np.float32([[1, 0, x_d],[0, 1, y_d]])

    return cv2.warpAffine(img, M, (cols, rows))

def jitter_rotation(img):
    """
    Randomly rotate the image between -15 and 15 degrees
    """
    rows, cols, channels = img.shape
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

    return cv2.warpAffine(img, M, (cols, rows))

def jitter_scale(img):
    """
    Randomly change the scale of the image in both x and y directions
    """
    x_scale = random.uniform(0.9, 1.1)
    y_scale = random.uniform(0.9, 1.1)
    rows, cols, channels = img.shape
    zeroed_image = np.zeros_like(img)
    return cv2.resize(img, (32,32), fx=x_scale, fy=y_scale, interpolation = cv2.INTER_CUBIC)

def blur(img):
    """
    Simple gaussian blur
    """
    return cv2.GaussianBlur(img,(3,3),0)

def jitter(img):
    """
    Combine all the jitter transforms together
    """
    return jitter_position(jitter_rotation(jitter_scale(img)))

```
I added 4 of these randomly jittered images to the training data.

```python
X_train_processed = []
for i, img in enumerate(X_train):
    # add original image with processing
    X_train_processed.append(process(img))
    y_train_processed.append(y_train[i])

    # add jittered data
    X_train_processed.append(process(jitter(img)))
    y_train_processed.append(y_train[i])

    X_train_processed.append(process(jitter(img)))
    y_train_processed.append(y_train[i])

    X_train_processed.append(process(jitter(img)))
    y_train_processed.append(y_train[i])

    X_train_processed.append(process(jitter(img)))
    y_train_processed.append(y_train[i])
```


Validation data can contain images that are slightly blurrier than others, so I added one more copy of each image
with a 3x3 Gaussian blur applied.

```python
    X_train_processed.append(process(blur(jitter(img))))
    y_train_processed.append(y_train[i])
```

A total of 5 additional images were created for each training data image.

Here are some random variations of one image, together with the final jittered copy (2nd from right)
and a copy with both blur and jitter (rightmost image):

![alt text][image3]

This transforms the original 34,799 training images to 208,794 images (1 original, 4 jittered, 1 jittered + blurred).

Because each iteration of the jitter processing applies each transform at random, each of the jittered copies of the image is highly likely to be different from the others.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My original attempts centered around a basic LeNet-5 architecture with 32x32x1 input and an output of 43 classes, but my testing with image processing and without modified training data did not yield very good validation accuracy.

After reading the [sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper multiple times,
what resonated most with me was having a two stage network, with the first stage looking at fine detail
and the second stage finding something more generalized. I tried to emulate aspects of this architecture.

I stuck with the same ReLU activation I had with LeNet-5 and added a random dropout of 0.3 to both convolutional layers
because this seemed to help validation accuracy change in a predictable manner.

```python
def TrafficSignClassifier(x): 
    mu = 0
    sigma = 0.1
    
    weights = {
    'w_conv1': tf.Variable(tf.truncated_normal([5, 5, num_channels, 108], mean=mu, stddev=sigma)),
    'w_conv2': tf.Variable(tf.truncated_normal([5, 5, 108, 200], mean=mu, stddev=sigma)),
    'w_fc1': tf.Variable(tf.truncated_normal(shape=(26168, 100), mean=mu, stddev=sigma)),
    'w_fc2': tf.Variable(tf.truncated_normal(shape=(100, n_classes), mean=mu, stddev=sigma)),
    }

    biases = {
    'b_conv1': tf.Variable(tf.zeros(108)),
    'b_conv2': tf.Variable(tf.zeros(200)),
    'b_fc1': tf.Variable(tf.zeros(100)),
    'b_fc2': tf.Variable(tf.zeros(n_classes)),
    }

    # 32x32x1 -> 28x28x108
    conv1 = tf.nn.conv2d(x, weights['w_conv1'], strides=[1, 1, 1, 1], padding='VALID') + biases['b_conv1']
    conv1 = tf.nn.relu(conv1)
    # dropout
    conv1 = tf.nn.dropout(conv1, keep_prob_cv)

    # 28x28x108 -> 14x14x108
    conv1 = maxpool2d(conv1)

    # 14x14x108 -> 10x10x200
    conv2 = tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 1, 1, 1], padding='VALID') + biases['b_conv2']
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob_cv)

    # 10x10x200 -> 5x5x200
    conv2 = maxpool2d(conv2, 2)

    # flatten to get a 21168 element hi res representation of layer 1 
    x_hi_res = flatten(conv1)
    # flatten to get a 5000 element lo res representation of layer 2
    x_lo_res = flatten(conv2)
    
    # 21168 + 5000 -> 26168
    fc1 = tf.concat([x_hi_res, x_lo_res], -1)
    fc1 = tf.nn.relu(fc1)

    # 26168 -> 100
    fc2 = tf.matmul(fc1, weights['w_fc1']) + biases['b_fc1']
    fc2 = tf.nn.relu(fc2)

    # 100 -> 43
    logits = tf.matmul(fc2, weights['w_fc2']) + biases['b_fc2']
    return logits, conv1, conv2
```

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image (processed Y channel)			| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x108 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x200	|
| RELU 					| 												|
| Max pooling			| 2x2 stride, outputs 5x5x200					|
| Flatten Convolution 1*| Outputs 21168									|
| Flatten Convolution 2*| Outputs 5000									|
| Concatenate flattened | Outputs 26168									|
| RELU 					| 												|
| Fully connected hidden| Outputs 100									|
| RELU 					| 												|
| Output				| Outputs 43									|
 
* The starred sections are merely descriptive and the actual layer is a concatenation of these

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the optimizer I stuck with the one from the LeNet-5 lab in the previous section:

```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

I initially started with a batch size of 128, before reducing it to 64, and then increasing it back to 128 when I realized that I wasn't seeing drastic differences between the 2. 

The learning rate I found that worked the best for a while was 0.0025, but in later iterations I reduced it to 0.0015 without much ill effect.

I initially trained for 10 epochs each time, but it seemed like accuracy was still increasing at the 10 epoch mark, so I increased this to 20 epochs. After more testing I settled on 25 epochs. I know there are smarter things that can be done with the termination condition, but given the speed of training, I did not experiment too much with this.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.947 
* test set accuracy of 0.909

Due to the high training set accuracy, I suspect I have overfit the training set, and if I were to work on this further, I would try to gain more validation set accuracy at the expense of training set accuracy.

* What was the first architecture that was tried and why was it chosen?

I started with LeNet-5 for the architecture. I was unable to tweak the input data enough to achieve a validation accuracy higher than the low 0.8xx range. This could have been because I didn't jitter the training data at the beginning, or my CLAHE transform was not very well-chosen at the time. Nevertheless, I started reading the sermanet paper more closely to try to reconstruct their methodology as best I could.

* What were some problems with the initial architecture?

The validation accuracy wasn't very high; it didn't seem like the model was big enough to adequately encode 43 different classes

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The key differences from the initial version ended up in the classifier section, where we concatenate the output of the first and second convolution layers, with the first convolution layer having higher resolution than the second.

The second difference is a 2-stage classifier, reducing 26168 outputs to 100 before reducing to 43 for the final classifier.

* Which parameters were tuned? How were they adjusted and why?

I played with the convolutional layers, starting with the parameters described in the sermanet paper (108/200) and trying others like 108/108. I also tried different filter sizes, such as 3x3 for the first layer and 5x5 for the second layer. Another thing I tried was using a different stride for the max pooling, such as a 4x4 stride for the second layer.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolutional layers help encode various features present on the various road signs. I used 2 convolutional layers to let the network remember additional representations of the data.

I also saw a noticeable increase in accuracy after increasing the depth of the convolutional layers. There was a noticeable jump in accuracy after I increased the layers to 108/200. Due to the speed of training I did not try more variations of these parameters..

I also used a couple dropout layers and they seemed to have helped training not get stuck in some local minimum.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![80 km/h][image4] ![Right-of-way at the next intersection][image5] ![Road work][image6] 
![End of all speed and passing limits][image7] ![Turn right ahead][image8]

These are sharp high quality images so technically they shouldn't be particularly hard to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image							        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| 80 km/h      							| 30 km/h   									| 
| Right-of-way at the next intersection | Dangerous curve to the left 					|
| Road work								| Road work										|
| End of all speed and passing limits	| End of all speed and passing limits			|
| Turn right ahead						| Turn right ahead      						|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The accuracy on the test set was ~90% so this isn't particularly great.

In both of the mistaken cases, the overall shape and style is similar, but the finest details are wrong. 80 and 30 are fairly similar numbers, and the "Dangerous curve" sign is a red triangle with an arrow in it similar to the "Right of way..." sign, the difference being the curve of the arrow inside the red triangle.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is very sure that this is a 30 km/h sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 30 km/h										| 
| .00000826     		| 50 km/h 										|
| .00000289				| 80 km/h										|
| .00000232	      		| 20 km/h						 				|
| .000000454		    | 70 km/h										|

For the second image, the network is fairly sure about it being a "Dangerous curve to the left" sign, which is inaccurate. I suspect this is because the angle of the photo distorts the sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .852         			| Dangerous curve to the left					| 
| .136     				| Traffic signals 								|
| .00989				| Keep left										|
| .00108	      		| General caution						 		|
| .000335			    | Yield											|


For the third image, we are extremely certain that it is a road work sign, which is correct


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Road work										| 
| .00000897     		| Slippery road 								|
| .000000266			| Dangerous curve to the right					|
| .000000211	  		| Children crossing						 		|
| .0000000106		    | Bumpy road									|

For the fourth, we are extremely sure about it being the "End of all speed and passing limits" sign, which is also correct

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| End of all speed and passing limits			| 
| .000387     			| End of no passing 							|
| .00000531				| End of speed limit (80km/h)					|
| .0000000258	  		| No passing					 				|
| .00000000000245		| Yield											|

For the fifth image we are extremely sure it is a "Turn right ahead" sign, which is correct

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Turn right ahead								| 
| .000331     			| Roundabout mandatory 							|
| .000000944			| Keep left										|
| .000000462			| Speed limit (100km/h)			 				|
| .000000379			| Priority road									|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is the output of the first convolutional layer (14x14x108) for the first downloaded image ("80 km/h"). It appears that some of these features look at overall shape (e.g. 91, 71) and some activate more strongly for details inside the sign, such as 24 or to a lesser extent 81. Looking at these features, it is possible I have more features than I need, or some of these features activate more strongly for other kinds of signs.

![Output of the first convolutional layer][image9]
