"""
input.shape = [batch_size, n_steps]     #[2,4]
embedding_input.shape = [batch_size, n_steps, embedding_size]  #[2,4,4]
lstm_output.shape = [batch_size, n_steps, lstm_size]    #最后一层神经元的输出
lstm_final_state: tuple
                  final_state = ((ci, hi) for i in range(n_layer))
                  ci.shape = [batch_size, lstm_size]
                  hi.shape = [batch_size, lstm_size]
initial_state.shape = final_state.shape
"""

import tensorflow as tf
import numpy as np

lstm_size = 5
keep_prob = 1.
n_layers = 3
batch_size = 2
vocab_size = 10
embedding_size = 4

# inputs
input = np.array([[1, 2, 5], [3, 4, 6]])
embedding = tf.get_variable(name="embedding",
                            shape=[vocab_size, embedding_size],
                            initializer=tf.random_uniform_initializer(-1.0, 1.0))
embedded = tf.nn.embedding_lookup(embedding, input)

# 创建单个cell
def get_a_cell(lstm_size, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop

# 堆叠多层神经元
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(lstm_size, keep_prob) for _ in range(n_layers)])

# 初始化神经元状态
initial_state = cell.zero_state(batch_size, tf.float32)

# 动态展开
lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=embedded, initial_state=initial_state)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    init=sess.run(initial_state)
    print(len(init))
    print(init)
    print("-------------------------")
    output = sess.run(lstm_outputs)
    print(output.shape)  # (2, 3, 5)   [batch_size, n_step, lstm_size]
    print(output)
    print("-------------------------")
    _ = sess.run(final_state)       # final_state = ((ci, hi) for i in range(n_layer))
    print(len(_))
    print(_)


