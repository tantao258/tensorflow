import tensorflow as tf

"""
矩阵修剪
strided_slice(
                input_,
                begin,
                end,
                strides=None,
                begin_mask=0,
                end_mask=0,
                ellipsis_mask=0,
                new_axis_mask=0,
                shrink_axis_mask=0,
                var=None,
                name=None
            )
begin，end和strides决定了input的每一维要如何剪切
"""
data = [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]]

x = tf.strided_slice(data, [0, 0, 0], [1, 1, 1])
y = tf.strided_slice(data, [0, 0, 0], [2, 2, 2], [1, 1, 1])
z = tf.strided_slice(data, [0, 0, 0], [2, 2, 2], [1, 2, 1])

with tf.Session() as sess:
    print("x--------------\n", sess.run(x))
    print("y--------------\n", sess.run(y))
    print("z--------------\n", sess.run(z))

"""
x--------------
 [[[1]]]
y--------------
 [[[1 1]
  [2 2]]

 [[3 3]
  [4 4]]]
z--------------
 [[[1 1]]

 [[3 3]]]

 x的输出为[[[1]]]，因为在每一维我们只截取[0,1)，因此只保留了[0,0,0]这里的元素，即1。
 y在每一位截取[0,2)，且步长为1，因此剩了8个元素。
 而z在第二维的步长是2，因此保留[0,0,0],[1,0,0],[1,0,1],[0,0,1]四个元素。
"""
