import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500
BATCH_SIZE=100

LEARNING_RATE_BASE=0.8
LEARING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

def inference(input_tensor,avg_class, weights1, biases1,weights2,biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)

        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1=tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1)
        )
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
#生成隐藏层
    weights1=tf.Variable(
        tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
#生成输出层
    weights2=tf.Variable(
        tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1)
    )
    biases2=tf.Variable(tf.constant(0,1,shape=[OUTPUT_NODE]))
#计算当前参数神经网络前向传播结果 滑动平均的类为null
    y=inference(x,None,weights1,biases1,weights2,biases2)
#定义存储训练轮数的变量 这里指定这个变量为不可训练的参数 trainable=false
    global_step=tf.Variable(0,trainable=False)
#给定训练轮数的变量 初始化滑动平均类
    # 可以加快训练早期变量的更新速度
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

#在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如 global_step)不需要
#tf.trainable_variable_averages
    variables_averages_op=variable_averages.apply(
        tf.trainable_variables()
    )
#计算使用滑动平均之后的前向传播结果 介绍过滑动平均不会改变变量本身的取值，维护一个影子变量 当需要使用这个滑动平均值时，需要明确调用average函数
    average_y=inference(
        x,variable_averages,weights1,biases1,weights2,biases2)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y,labels=tf.argmax(y_,1)
    )

    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization=regularizer(weights1)+regularizer(weights2)
    loss=cross_entropy_mean+regularization
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARING_RATE_DECAY
    )

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean,global_step=global_step)

    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed={x: mnist.validation.images,
                       y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,
                   y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                test_acc=sess.run(accuracy,feed_dict=test_feed)
                print("After %d training step(s),validation accuracy"
                       "using average model is %g,test accuracy using averag model is %g"%(i,validate_acc,test_acc))
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average"
              "model is %g"% (TRAINING_STEPS,test_acc))

def main(argv=None):
    mnist=input_data.read_data_sets("D:\\program\\python\\learn_tensorflow.git\\trunk\\MNIST",one_hot=True)
    train(mnist)

if __name__=='__main__' :
    tf.app.run()