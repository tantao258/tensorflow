"""
outputs = activation(inputs * kernel + bias)
"""

import numpy as np
import tensorflow as tf

# batch_size = 5
# ones = tf.ones([batch_size, 20])
# logits = tf.layers.dense(ones, 10)
# print(logits)
# for var in tf.trainable_variables():
#     print(var)
# # Tensor("dense/BiasAdd:0", shape=(5, 10), dtype=float32)
print("-----------------------我是漂亮的分割线----------------------------")

# batch_size = 5
# ones = tf.ones([batch_size, 8, 20])
# logits = tf.layers.dense(ones, 10)
# print(logits)
# for var in tf.trainable_variables():
#     print(var)
# # Tensor("dense_1/BiasAdd:0", shape=(5, 8, 10), dtype=float32)
print("-----------------------我是漂亮的分割线----------------------------")
batch_size = 5
ones = tf.ones([batch_size, 6, 8, 20])
logits = tf.layers.dense(ones, 10)
print(logits)
for var in tf.trainable_variables():
    print(var)
# Tensor("dense_2/BiasAdd:0", shape=(5, 6, 8, 10), dtype=float32)
