import tensorflow as tf
'''
[
			[[0], [1], [1], [0], [0], [0], [1], [1]],
			[[0], [1], [1], [0], [0], [0], [1], [1]],
			[[0], [0], [0], [0], [0], [0], [0], [0]],
			[[1], [1], [0], [0], [1], [1], [0], [0]],
			[[1], [1], [0], [0], [1], [1], [0], [0]],
			[[0], [0], [0], [0], [0], [0], [0], [0]],
			[[0], [0], [0], [0], [0], [0], [0], [0]],
			[[0], [0], [0], [0], [0], [0], [0], [0]]
		]
'''

initial_input = tf.Variable(
	[
		[
			[[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]],
			[[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]],
			[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
			[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
			[[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]],
			[[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]],
			[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
			[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
		],
		[
			[[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]],
			[[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]],
			[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
			[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
			[[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]],
			[[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]],
			[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
			[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
		],

	], 
	dtype='float32')


input_stacked = tf.transpose(initial_input, [0, 3, 1, 2])
print input_stacked
#Index 3 and 5 are our window sizes
#Index 1 is our channel
#Index 0 is our batch
#Index 2 and 4 are our resized shape
input_windowed = tf.transpose(tf.reshape(input_stacked, [2, 4, 4, 2, 4, 2]), [0, 1, 2, 4, 3, 5])
print input_windowed

sum_input = tf.reduce_sum(tf.reduce_sum(input_windowed, 5), 4)
print sum_input

# Batch
# Channels
# Rest
flattened_sum = tf.reshape(sum_input, [2, 4, -1])
print flattened_sum

# Window size * 2
argmax_flat = tf.nn.top_k(flattened_sum, k=4, sorted=False)
max_flat_indeces = argmax_flat[1] 
print max_flat_indeces

topk_indeces = tf.nn.top_k(max_flat_indeces, k=4)
topk_indeces_values = tf.reverse(topk_indeces[0], [2])
print topk_indeces_values

# Resized Shape 
x_indeces = topk_indeces_values % 4
y_indeces = topk_indeces_values // 4
print x_indeces
print y_indeces

indeces = tf.stack([y_indeces, x_indeces], axis=-1)
print indeces

# Resized Shape
v = tf.tile([[[0]], [[1]]], [1, 4, 4])

# Window shape * 2
# Resized Shape
w = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, 4), axis=-1), [2, 4]), [2, 4, 4])
z = tf.stack([v, w], axis=-1)

q = tf.concat([z, indeces], axis=-1)

# 0 - Batch
# 1 - Channel
# 2 - Y / Rows
# 3 - X / Cols
#windows = tf.gather_nd(input_windowed, [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 2], [0, 3, 1, 1]])
windows = tf.gather_nd(input_windowed, q)
print windows
# Who fucking knows how this works
# Reverse of the above one
l = tf.reshape(windows, [2, 4, 2, 2, 2, 2])
m = tf.transpose(l, [0, 1, 2, 4, 3, 5])
o = tf.reshape(m, [2, 4, 4, 4])
# Batch
# Channels
# Resized Shape
windows_reshaped = tf.reshape(windows, [2, 4, 4, 4])
windows_transposed = tf.transpose(windows_reshaped, [0, 3, 1, 2])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run([init_op])

	print input_stacked.eval()
	#print input_windowed.eval()
	#print sum_input.eval()
	#print flattened_sum.eval()
	#print max_values.eval()
	print max_flat_indeces.eval()
	print topk_indeces_values.eval()
	#print x_indeces.eval()
	#print y_indeces.eval()
	#print indeces.eval()
	#print z.eval()
	print q.eval()
	print windows.eval()
	#print l.eval()
	#print m.eval()
	print o.eval()
	#print windows_reshaped.eval()
	#print windows_transposed.eval()

