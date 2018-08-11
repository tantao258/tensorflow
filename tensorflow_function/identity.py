"""
identity(input, name=None)
Return a tensor with the same shape and contents as input
"""
import tensorflow as tf

a = [[1, 2, 3, 4], [4, 6, 8, 0]]
b = tf.identity(a, name="b")

with tf.Session() as sess:
    print(b)
    print(sess.run(b))

# ---------------------
# Tensor("b:0", shape=(2, 4), dtype=int32)
# [[1 2 3 4]
#  [4 6 8 0]]
