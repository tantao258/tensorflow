"""
sequence_mask(
                lengths,          lengths是一个一维数组,代表每一个sequence的长度
                maxlen=None,
                dtype=tf.bool,
                name=None
)
返回的是一个mask的张量，张量的维数是：(len(lengths), maxlen)

"""

import tensorflow as tf

lengths = [1, 3, 2]
output = tf.sequence_mask(lengths=lengths, maxlen=5)


lengths1 = [[1, 3], [2, 0]]
output1 = tf.sequence_mask(lengths=lengths1, maxlen=5)


with tf.Session() as sess:
    print(sess.run(output))
    print("--------------------")
    print(sess.run(output1))


"""
[[ True False False False False]   1
 [ True  True  True False False]   3
 [ True  True False False False]]  2
--------------------
[[[ True False False False False]  1
  [ True  True  True False False]] 3

 [[ True  True False False False]   2
  [False False False False False]]] 0

Process finished with exit code 0
"""