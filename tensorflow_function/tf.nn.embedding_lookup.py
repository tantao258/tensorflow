"""
tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引，其他的参数不介绍。
embedding.shape = [vocab_size, embedding_size]
embedded.shape = input.shape + [embedding_size] 构成三维数组
note:
    tensorflow's embedding matrix 没有特定的关联意义。
"""

import tensorflow as tf
import numpy as np

embedding_size = 5
vocab_size = 10

input = np.array([[1, 2], [3, 4]])
embedding = tf.get_variable(name="embedding",
                            shape=[vocab_size, embedding_size],
                            initializer=tf.random_uniform_initializer(-1.0, 1.0))
embedded = tf.nn.embedding_lookup(embedding, input)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print("----------Input--------------")
    print(input)
    print("--------embedding-------------")
    print(sess.run(embedding))
    print("--------embedded--------------")
    print(sess.run(embedded))