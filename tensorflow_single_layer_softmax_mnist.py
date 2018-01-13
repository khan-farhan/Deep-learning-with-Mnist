import tensorflow as tf
from tensorflow_first import output
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.nn.python.ops import cross_entropy
from tensorflow_mnist_ffn import batch_size


mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

 ################################# Variable Initialization ###########################################################
no_of_nodes = 10
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,10])


################################## Graph creation ###################################################################

'''
This function creates the neural network
'''

def net(x):
    hidden_aka_output_layer = {'weights' : tf.Variable(tf.zeros([784,10])) , 'biases' : tf.Variable(tf.zeros([10])) }
    
    output = tf.add(tf.matmul(x,hidden_aka_output_layer['weights']) , hidden_aka_output_layer['biases'])
    
    return output
    
'''
This function is for training and testing the model
'''
def train_neural_net(x):
    
    prediction = net(x)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) # # the cost function
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)   ## optimization function
    
    no_epochs = 15            ## no of training repetition

#### starting the session
   
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epochs in range(no_epochs):
            epoch_loss =  0
            
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x , epoch_y = mnist.train.next_batch(batch_size) 
                _, c = sess.run([optimizer,cost], feed_dict = { x : epoch_x , y : epoch_y }) # feeding the data with feed_dict: these x and y are the one which are defined globally
                epoch_loss += c
                
            print("After" , epochs ,"epochs the loss is " , epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        print("Accuracy of the model is :" , accuracy.eval({x : mnist.test.images , y : mnist.test.labels}))
        
        
        
train_neural_net(x)
    
    
    
    
