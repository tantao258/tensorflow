import tensorflow as tf


# ------------------------------------------------- save model --------------------------------------------------
x1 = tf.placeholder(dtype=tf.float32, name="x1")
x2 = tf.placeholder(dtype=tf.float32, name="x2")


w1 = tf.Variable(tf.constant(2.0, shape=[1]), name="w1")
w2 = tf.Variable(tf.constant(3.0, shape=[1]), name="w2")

y = x1 * w1 + x2 * w2

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y, feed_dict={x1: 2, x2: 3}))
    saver.save(sess, "./checkpoint/model.ckpt")




# -------------------------------------------------restore model ----------------------------------------------------
# x1 = tf.placeholder(dtype=tf.float32, name="x1")
# x2 = tf.placeholder(dtype=tf.float32, name="x2")
#
#
# w1 = tf.Variable(tf.constant(2.0, shape=[1]), name="w1")
# w2 = tf.Variable(tf.constant(3.0, shape=[1]), name="w2")
#
# y = x1 * w1 + x2 * w2
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     saver.restore(sess, "./checkpoint/model.ckpt")
#     print(sess.run(y, feed_dict={x1: 2, x2: 3}))