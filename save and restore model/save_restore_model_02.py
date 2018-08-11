import tensorflow as tf

# ------------------------------------------------- save model --------------------------------------------------
# x1 = tf.placeholder(dtype=tf.float32, name="x1")
# x2 = tf.placeholder(dtype=tf.float32, name="x2")
#
# w1 = tf.Variable(tf.constant(2.0, shape=[1]), name="w1")
# w2 = tf.Variable(tf.constant(3.0, shape=[1]), name="w2")
#
# y = x1 * w1 + x2 * w2
#
# saver = tf.train.Saver()
# tf.add_to_collection('result', y)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(y, feed_dict={x1: 2, x2: 3}))
#     saver.save(sess, "./checkpoint/model.ckpt")

print("---------------------------------------------我是漂亮的分割线-----------------------------------------------")

# -------------------------------------------------restore model ----------------------------------------------------
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./checkpoint/model.ckpt.meta")
    saver.restore(sess, "./checkpoint/model.ckpt")
    graph = tf.get_default_graph()

    # x1 = graph.get_operation_by_name("x1").outputs[0]
    # x2 = graph.get_operation_by_name("x2").outputs[0]
    # y = tf.get_collection('result')[0]
    x1 = graph.get_tensor_by_name("x1:0")
    x2 = graph.get_tensor_by_name("x2:0")
    y = graph.get_tensor_by_name("add:0")

    print(sess.run(y, feed_dict={x1: 2, x2: 3}))


