import tensorflow as tf

session = tf.Session()

# A placeholder is a promise to provide a value later
first_var = tf.placeholder(tf.float32)
second_var = tf.placeholder(tf.float32)

adder_node = first_var + second_var

# sess.run(fetches, feed_dict=None, options=None, run_metadata=None)
# feed_dict is used to override the values of the tensors in the graph
print(session.run(adder_node, {first_var: 3.0, second_var: 4.0}))
print(session.run(adder_node, {first_var: [3,3.5], second_var: [1,0]}))
print(session.run(adder_node, {first_var: [[1,2,3],[4,5,6]], second_var: [2,2,2]}))

# Add operations to make the computation graph more complex
add_then_triple = adder_node * 3
print(session.run(add_then_triple, {first_var: 2, second_var: 3}))

# Variables allow training, but are not initialized at first
# Create a linear model
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# Initialize all global variables and feed_dict the x values
init = tf.global_variables_initializer()
session.run(init)
print(session.run(linear_model, {x: [1,2,3,4,5]}))

# Write the loss function, which measures how far apart the current model is from the data
# Use standard loss model for linear regression, the sum of the squares of both deltas
# (linear_model - y) gives the error
# tf.square() squares it
# tf.reduce_sum() computes the sum of elements across dimensions of a tensor
y = tf.placeholder(tf.float32)
error = linear_model - y
squared_deltas = tf.square(error)
loss = tf.reduce_sum(squared_deltas)
print(session.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# You can change variable values after initialization using tf.assign(var, val)
# Changing this will result in a loss of 0.0, as W=-1 and b=1 are 'perfect' values
fixed_W = tf.assign(W, [-1.0])
fixed_b = tf.assign(b, [1.0])
session.run([fixed_W, fixed_b])
print(session.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))