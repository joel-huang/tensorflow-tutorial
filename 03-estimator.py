import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Create an estimator that does linear regression
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# Set up the training set
train_x = np.array([1.,2.,3.,4.])
train_y = np.array([0.,-1.,-2.,-3.])

# Set up the testing set
test_x = np.array([2.,5.,8.,1.])
test_y = np.array([-1.01,-4.1,-7,0.])

# Build the input functions
# Specify the batch size and the number of batches (num_epochs)
# numpy_input_fn(dict of numpy array, numpy array, batch size, num_epochs, shuffle)
# Returns the input function that would feed dict of numpy arrays into the model

input_function = tf.estimator.inputs.numpy_input_fn(
        {"x": train_x}, train_y, batch_size=4, num_epochs=None, shuffle=True)

training_input_function = tf.estimator.inputs.numpy_input_fn(
        {"x": train_x}, train_y, batch_size=4, num_epochs=1000, shuffle=False)

testing_input_function = tf.estimator.inputs.numpy_input_fn(
        {"x": test_x}, test_y, batch_size=4, num_epochs=1000, shuffle=False)

# Train the estimator
estimator.train(input_fn=input_function, steps=1000)

# Evaluate the estimator
train_result = estimator.evaluate(input_fn=training_input_function)
eval_result = estimator.evaluate(input_fn=testing_input_function)

print("training results: %r"%train_result)
print("testing results: %r"%eval_result)
