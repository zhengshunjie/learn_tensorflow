import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)


#每个批次大小
batch_size=100
n_batch=mnist.train.num_examples//batch_size

#初始化偏置
def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)

#初始化偏置
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    #x input tensor of shape '[batch,in_height,in_width,in_channels]'
    #W filter/kernel tensor of shape [filter_height ,filter_width,in_channels,out_channels]
    #'strides[0] = strides[3] = 1' shrides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding:
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
 
with tf.name_scope('input'):
    #定义两个placeholder
    x=tf.placeholder(tf.float32,[None,784])#28*28
    y=tf.placeholder(tf.float32,[None,10])
    with tf.name_scope('x_image'):
        #改变x的格式转为4D的向量[batch，in_height,in_width,in_channels]
        x_image=tf.reshape(x,[-1,28,28,1])
with tf.name_scope('Convl'):
    #初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1=weight_variable([5,5,1,32])#5*5采样窗口，32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1=bias_variable([32])#每一个卷积核一个偏置值
    with tf.name_scope('relu'):
        #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
        h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    with tf.name_scope('h_pool1'):
        h_pool1=max_pool_2x2(h_conv1)

#初始化第二个卷积层的全值和偏执
with tf.name_scope('Conv1'):
    with tf.name_scope('W_conv2'):
        W_conv2=weight_variable([5,5,32,64])#5*5采样窗口，32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2=bias_variable([64])#每一个卷积核一个偏置值
    with tf.name_scope('relu'):
        #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
        h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    with tf.name_scope('h_pool2'):
        h_pool2=max_pool_2x2(h_conv2)

#初始化第一个全连接层的权值
W_fcl=weight_variable([7*7*64,1024])#上一层有7*7*64个神经元，全连接层有1024个神经元
b_fcl=bias_variable([1024])#1024个节点

#把池化层2的输出扁平化为1维
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
#求第一个全连接层的输出
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl)+b_fcl)

#keep_prob用来表示神经元的输出概率
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#初始化第二个全连接层
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

#计算输出
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#交叉代价函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用 AdamOptimizer进行优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个bool列表中
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Iter "+str(epoch)+", Testing Accuracy="+str(acc))