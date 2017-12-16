import tensorflow as tf

# Steps:
# 1. Initialize the variables W, x, b
# 2. Create a loss function for linear regression
# 3. Train a gradient descent optimizer
# 4. Evaluate the accuracy

# Step 1: Initialize the variables
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
model = W*x + b

# Step 2: Create the loss function
y = tf.placeholder(tf.float32)
error = model - y
loss = tf.reduce_sum(tf.square(error))

# Step 3: Train the gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

train_x = [1,2,3,4]
train_y = [0,-1,-2,-3]

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

for i in range(2000):
    session.run(train, {x: train_x, y: train_y})
    current_W, current_b, current_loss = session.run([W,b,loss], {x: train_x, y: train_y})
    print("W: %s b: %s loss: %s"%(current_W, current_b, current_loss))
