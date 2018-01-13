import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_first import output
from tensorflow.python.ops.variables import initialize_all_variables
from nltk.chunk.util import accuracy
from cryptography.hazmat.primitives import padding
from bokeh import __main__


 ################################# Variable Initialization ###########################################################
 
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

no_classes = 10
batch_size = 100

x = tf.placeholder('float',[None,784])       ##### none represents anyisize corresponding to the batch size
y = tf.placeholder('float',[None,10])
keep_prob = tf.placeholder('float')

################################## Graph creation ###################################################################




def convolution(x,w):                   
                                ##size of stride = 1
    return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME')



def max_pooling(x):
                                #### size of map 2*2         size of stride = 2
    return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1] , padding = 'SAME')





'''
This function creates the neural network

Here 2 layer of convolution are used. In the first layer we are convolving with a filter of 5*5 and we using a stack of 32 
filters. The image used in this layer is a single 2D image of size 28*28*1. Maxpooling is used to simplify the output by using
2*2 map of stride size 1.

The image in the 2nd layer becomes a 14*14*32 image. In this layer convlution is done using a 5*5 filters of stack size 64.
Maxpooling is done in the similar manner.

The image in the third layer becomes 7*7*64. Fully connected layer involves 1024 units. 

before the output layer dropout is used to reduce the overfitting and computation time




'''
def CNN(x):
    
    weights = {'W_conv1' : tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2' : tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc'    : tf.Variable(tf.random_normal([7*7*64,1024])),
               'W_out'   : tf.Variable(tf.random_normal([1024,no_classes]))
               }
    
    
    biases = {'B_conv1' : tf.Variable(tf.random_normal([32])),
              'B_conv2' : tf.Variable(tf.random_normal([64])),
              'B_fc'    : tf.Variable(tf.random_normal([1024])),
              'B_out'   : tf.Variable(tf.random_normal([no_classes]))
               }
   
    
    x = tf.reshape(x, [-1, 28, 28, 1])       #### reshaping the flattened image to size of 28*28 format is [batchsize, img_len, img_breath, no_of_channels]
  ################################## First layer ########################################
    
    conv1 = convolution(x,weights['W_conv1'])
    conv1 = tf.nn.relu(conv1 + biases['B_conv1'])
    maxpool1 = max_pooling(conv1)
  
  ################################## second layer #######################################  
  
    conv2 = convolution(maxpool1,weights['W_conv2'])
    conv2 = tf.nn.relu(conv2 + biases['B_conv2'])
    maxpool2 = max_pooling(conv2)
   
   ################################ Fully connected layer ########################### 
   
    fc = tf.reshape(maxpool2, [-1, 7*7*64])      #### reshaping the 7*7*7 image to flattened
    fc = tf.nn.relu(tf.matmul(fc , weights['W_fc']) + biases['B_fc'])
    
    
    ############################### Dropout #######################################
    
    drop = tf.nn.dropout(fc, keep_prob)
    
    ############################## output Layer #######################################
    
    output = tf.matmul(drop ,weights['W_out']) + biases['B_out']
    
    return output


'''
This function is for training and testing the model
'''

def train_CNN(x):
    prediction = CNN(x)
  
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) ) ## the cost function
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)                                      ## the optimizer function
    
    no_epochs = 15            ## no of training repetition

#### starting the session
   
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epochs in range(no_epochs):
            epoch_loss =  0
            
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x , epoch_y = mnist.train.next_batch(batch_size) 
                _, c = sess.run([optimizer,cost], feed_dict = { x : epoch_x , y : epoch_y, keep_prob: 1.0 }) # feeding the data with feed_dict: these x and y are the one which are defined globally
                epoch_loss += c
                
            print("After" , epochs ,"epochs the loss is " , epoch_loss)
            
## Training  ends  here
              
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        print("Accuracy of the model is :" , accuracy.eval({x : mnist.test.images , y : mnist.test.labels, keep_prob: 1.0}))
        
        



if __name__ == '__main__':
    
    train_CNN(x)




