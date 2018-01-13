import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_first import output
from tensorflow.python.ops.variables import initialize_all_variables
from nltk.chunk.util import accuracy


 ################################# Variable Initialization ###########################################################
 
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

no_nodes_hl1 = 600
no_nodes_hl2 = 600
no_nodes_hl3 = 600

no_classes = 10
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,10])

################################## Graph creation ###################################################################

'''
This function creates the neural network
'''

def neural_net(x):
    
    hidden_l_1 = {'weights' : tf.Variable(tf.random_normal([784,no_nodes_hl1])), 'biases' : tf.Variable(tf.random_normal([no_nodes_hl1]))}
    hidden_l_2 = {'weights' : tf.Variable(tf.random_normal([no_nodes_hl1,no_nodes_hl2])), 'biases' : tf.Variable(tf.random_normal([no_nodes_hl2]))}
    hidden_l_3 = {'weights' : tf.Variable(tf.random_normal([no_nodes_hl2,no_nodes_hl3])), 'biases' : tf.Variable(tf.random_normal([no_nodes_hl3]))}
    
    output_l = {'weights' : tf.Variable(tf.random_normal([no_nodes_hl3,no_classes])), 'biases' : tf.Variable(tf.random_normal([no_classes]))}

    activation_l1 = tf.add(tf.matmul(x , hidden_l_1['weights']), hidden_l_1['biases'])
    activation_l1 = tf.nn.relu(activation_l1)
    
    activation_l2 = tf.add(tf.matmul(activation_l1 ,hidden_l_2['weights']), hidden_l_2['biases'])
    activation_l2 = tf.nn.relu(activation_l2)
    
    activation_l3 = tf.add(tf.matmul( activation_l2, hidden_l_3['weights']), hidden_l_3['biases'])
    activation_l3 = tf.nn.relu(activation_l3)
    
    output = tf.add(tf.matmul(activation_l3,output_l['weights']) , output_l['biases'])
    
    
    return output


'''
This function is for training and testing the model
'''

def train_neural_net(x):
    prediction = neural_net(x)
    
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
                _, c = sess.run([optimizer,cost], feed_dict = { x : epoch_x , y : epoch_y }) # feeding the data with feed_dict: these x and y are the one which are defined globally
                epoch_loss += c
                
            print("After" , epochs ,"epochs the loss is " , epoch_loss)
            
## Training  ends  here
              
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        print("Accuracy of the model is :" , accuracy.eval({x : mnist.test.images , y : mnist.test.labels}))
        
        
        
train_neural_net(x)




