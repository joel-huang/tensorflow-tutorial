import matplotlib.pyplot as plot
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST dataset (handwritten digits)
# MNIST dataset has 70,000 images and labels
# with 3 mutually exclusive subsets:
# Training: 55000, Test: 10000, Validation: 5000
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# Specify data dimensions
# MNIST images are 28x28
img_size = 28

# Images are stored in 1D arrays of this length
img_size_flat = img_size * img_size

# Tuple with H and W of images, used to reshape arrays
img_shape = (img_size, img_size)

# Number of classes
num_classes = 10

# Start making the tensorflow placeholders and variables
    
# x will store the images in flattened form
# tf.placeholder(datatype, tensor size)
# number of cols in the tensor is == flattened imgsize
x = tf.placeholder(tf.float32, [None, img_size_flat])

# y_true will store the true classes of the images
# Also provide a one-hot encoded version(y_true_cls)
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

# Build the Variables to be optimized
# Weights - what is the matrix shape to multiply by x?
# Weights will be 784*10 matrix, biasses is a 1*10
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

# Logits will be a matrix with num_images rows
# and num_classes columns, where the (i,j)-th element
# is how likely the i-th image is of the j-th class.
logits = tf.matmul(x, weights) + biases

# Obtain the softmax of the xW+b matrix
# Use argmax to pick out the class numbers for each image
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

# So now, we need to compare the y_pred matrix with the y_true
# matrix. Use tf's built in cross_entropy() function:
# Cross entropy is zero when y_pred == y_true. So we want to
# MINIMIZE cross entropy by changing the weights and biases of the
# linear model in order to get the closest to zero value.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_true)

# The average of the cross entropy values for each image is taken,
# this is the function to be minimized.
cost = tf.reduce_mean(cross_entropy)

# Create a gradient descent optimizer at a step-size of 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Measures of performance
# Compare the predicted and actual class matrices
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# Specify how to calculate accuracy of the model by finding the mean of all
# entries in correct_prediction. To do this, we have to typecast 
# correct_prediction to tf.float32, which turns True to 1.0 and False to 0.0.
# Then we can carry out the tensor reduction.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

# Use a small batch size in each iteration of the optimizer (stochastic
# gradient descent). In each iteration, 100 new images are selected from
# the training set and tf executes the optimizer using those samples.
batch_size = 100

def optimize(num_iterations):
    for i in range(num_iterations):
        # Load the new batch of training data into x_batch and y_true_batch
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        
        # Create the dict of training data to be used in feed_dict
        train_dict = {x: x_batch, y_true: y_true_batch}
        
        # Run the optimizer (gradient descent optimizer)
        session.run(optimizer, feed_dict=train_dict)

# Lastly, we will need printing functions to print performance information
# Now, we deal with the testing set.

# Labels are one-hot encoded, the labels are converted
# from a single number to a vector. For example if the
# class is 7, then the one-hot encoded vector will be
# [0,0,0,0,0,0,0,1,0,0].

# Convert classes from one-hot to integer numbers
# np.argmax() searches for the "1" among the 0s,
# and returns the list index of that "1" value.
cls_list = []
for label in data.test.labels:
    # The position of the "1" in each list is
    # appended to the cls_list
    cls_list.append(label.argmax())
    
# Array-ify it with np
cls_array = np.array(cls_list)

# Create a dictionary with the test set information
test_dict = {x: data.test.images,
             y_true: data.test.labels, # The one-hot vector labels
             y_true_cls: cls_array}

# Run the test_dict into the accuracy function and print the result
def print_accuracy():
    acc = session.run(accuracy, feed_dict=test_dict)
    print("Accuracy: {0:.1%}".format(acc))
    
# Use scikit-learn's confusion matrix
def print_confusion_matrix():
    predicted_classes = session.run(y_pred_cls, feed_dict=test_dict)
    true_classes = cls_array
    cm = confusion_matrix(y_true=true_classes,
                          y_pred=predicted_classes)
    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plot.imshow(cm, interpolation='nearest', cmap=plot.cm.Blues)

    # Make various adjustments to the plot.
    plot.tight_layout()
    plot.colorbar()
    tick_marks = np.arange(num_classes)
    plot.xticks(tick_marks, range(num_classes))
    plot.yticks(tick_marks, range(num_classes))
    plot.xlabel('Predicted')
    plot.ylabel('True')
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plot.show()

# Plot examples of images that have been mis-classified
def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=test_dict)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_array[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# Plot weights of the model        
def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plot.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plot.show()

# Helper function to plot 9 images in a 3x3 grid
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plot.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "true: {0}".format(cls_true[i])
        else:
            xlabel = "Actual: {0}, Predicted: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plot.show()
    
# We are ready to run! Let's go!
print_accuracy() # 9.8%
plot_example_errors() # Everything predicted to be 0

# This is because the model has only been initialized and not optimized.
# 9.8% of the images happen to be 0.

# DONE!

# Exercises
# Change the learning-rate for the optimizer. - seems to be not much difference
# Change the optimizer to e.g. AdagradOptimizer or AdamOptimizer. - AdamOptimizer has troubles with 2s and 5s
# Change the batch-size to e.g. 1 or 1000. - tried batch size 10000, really slowed things down.
# How do these changes affect the performance? - answered above
# Do you think these changes will have the same effect (if any) on other classification problems and mathematical models? - probably not?
# Do you get the exact same results if you run the Notebook multiple times without changing any parameters? Why or why not? - no. each batch size of 100 is selected at random from the test set, i think.
# Change the function plot_example_errors() so it also prints the logits and y_pred values for the mis-classified examples. - not done
# Use sparse_softmax_cross_entropy_with_logits instead of softmax_cross_entropy_with_logits. This may require several changes to multiple places in the source-code. Discuss the advantages and disadvantages of using the two methods. - not done
# Remake the program yourself without looking too much at this source-code. - not done
# Explain to a friend how the program works. - not done

