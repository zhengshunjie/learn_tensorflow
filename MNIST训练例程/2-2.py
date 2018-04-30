import tensorflow as tf

x=tf.Variable([1,2])
a=tf.constant([3,3])
#增加一个减法op
sub =tf.subtract(x,a)
#增加一个加法op
add = tf.add(x,sub)
#全局变量初始化  变量一定要初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #先进行初始化
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

#创建一个变量初始化为0
state = tf.Variable(0,name = 'counter')
#创建一个op 作用是使state加1
new_value=tf.add(state,1)
#tensorflow中赋值需要使用assign函数
update=tf.assign(state,new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))