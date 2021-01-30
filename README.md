
# Puzzle
The objective of this neural network is to solve the following puzzle:    
1 + 4 = 5    
2 + 5 = 12  
3 + 6 = 21  
8 + 11 = ? 


```python
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# Working sets of hyperparameters
# 200 epochs - step size 0.1 - mse loss - sigmoid activations - 1 hidden layer with 3 neurons
# 400 epochs - step size 2e-2 - mse loss -  tanh activations - 1 hidden layer with 3 neurons
# 400 epochs - step size 2e-2 - Huber loss -  tanh activations - 1 hidden layer with 2 neurons
EPOCHS = 500

rate = 2e-2
global_step = tf.Variable(0, trainable=False)
# rate = tf.train.exponential_decay(rate, global_step, 5 * 3, 0.9, staircase=True)

X_train = np.asarray([[1, 4], [2, 5], [3, 6]])
Y_train = np.asarray([5, 12, 21])
```



### Input
Two numbers

### Architecture

**Layer 1: Fully Connected.** 

**Activation.** Tanh

**Layer 2: Fully Connected.** 

### Output
1 Number


```python
def nn(x):    
    t1 = tf.layers.dense(x, 10, name="t1", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.tanh)
    t2 = tf.layers.dense(t1, 10, name="t2", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
    t3 = tf.layers.dense(t2, 10, name="t3", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
    
    r1 = tf.layers.dense(x, 10, name="r1", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
    r2 = tf.layers.dense(r1, 10, name="r2", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
    r3 = tf.layers.dense(r2, 10, name="r3", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
    
    augment = tf.concat([t3, r3, x], 1, name="aug")
    outp = tf.layers.dense(augment, 1, name="output", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    return outp
```


```python
x = tf.placeholder(tf.float32, (None, 2), name="input_data")
# x = tf.placeholder(tf.float32, (None), name="input_data")
y = tf.placeholder(tf.int32, (None), name='true_labels')
```


```python
logits = nn(x)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
```


```python
# new_logits = logits - rate * acc_mean
loss_operation = tf.losses.huber_loss(labels=y, predictions=logits)
# loss_operation = tf.losses.mean_squared_error(labels=y, predictions=logits)
```


```python
# grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
training_operation = optimizer.minimize(loss_operation, global_step=global_step)
```


```python
def evaluate(a, b):
    sess = tf.get_default_session()
    answer = sess.run([logits], feed_dict={x:[[a, b]]})
#     answer = sess.run([logits], feed_dict={x:[a]})
    return round(answer[0][0][0])
```


```python
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
      X_train, Y_train = shuffle(X_train, Y_train)

      for cnt in range(3):
#           inp = np.reshape(X_train[cnt], [1,1])
          inp = np.expand_dims(X_train[cnt], 0)
        
          _ = sess.run([training_operation], #, tf.get_default_graph().get_tensor_by_name("aug:0")],
#                                       feed_dict={x: inp, y: Y_train[cnt]})
                                      feed_dict={x: np.expand_dims(X_train[cnt, :], 0), y: Y_train[cnt]})

    print(evaluate(1, 4))
#     answer = logits.eval(feed_dict={x:[[2, 5]]})
    print(evaluate(2, 5))
#     answer = logits.eval(feed_dict={x:[[3, 6]]})
    print(evaluate(3, 6))
#     answer = sess.run([logits], feed_dict={x:[[2, 3]]})
#     print(round(answer[0][0][0]))
    print(evaluate(2, 3))
    print(evaluate(8, 11))
    print()
```

    Training...
    
    5.0
    12.0
    21.0
    13.0
    36.0
    

