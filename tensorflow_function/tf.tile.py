"""
tile(
    input,
    multiples,
    name=None
)
tf.tile主要的功能就是在tensorflow中对矩阵进行自身进行复制的功能，比如按行进行复制，或是按列进行复制
"""

import tensorflow as tf
a = tf.constant([[1, 2], [2, 3], [3, 4]], dtype=tf.float32)
tile_a_1 = tf.tile(a, [1, 2])       #列 1个a, 行 2个a
tile_a_2 = tf.tile(a, [2, 1])       #列 2个a, 行 1个a
tile_a_3 = tf.tile(a, [2, 2])       #列 2个a, 行 2个a

with tf.Session() as sess:
    print("tile_a_1------------\n", sess.run(tile_a_1))
    print("tile_a_2------------\n", sess.run(tile_a_2))
    print("tile_a_3------------\n", sess.run(tile_a_3))

"""
tile_a_1------------
 [[1. 2. 1. 2.]
 [2. 3. 2. 3.]
 [3. 4. 3. 4.]]
tile_a_2------------
 [[1. 2.]
 [2. 3.]
 [3. 4.]
 [1. 2.]
 [2. 3.]
 [3. 4.]]
tile_a_3------------
 [[1. 2. 1. 2.]
 [2. 3. 2. 3.]
 [3. 4. 3. 4.]
 [1. 2. 1. 2.]
 [2. 3. 2. 3.]
 [3. 4. 3. 4.]]

Process finished with exit code 0
"""
