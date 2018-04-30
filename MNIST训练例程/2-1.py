import tensorflow as tf
#一行两列
m1=tf.constant([[3,3]])
#两行一列
m2=tf.constant([[2],[3]])
#创建矩阵乘法op
product=tf.matmul(m1,m2)
print(product)
#定义绘画 会有默认图
sess=tf.Session()
#执行这句之前的才会执行

with tf.Session() as sess:
    result=sess.run(product)
    print(product)