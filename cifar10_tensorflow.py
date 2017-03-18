import tensorflow as tf

initial_input = tf.Variable(
	[
		[
			[[1], [1], [0], [0], [0], [1], [1], [0]],
			[[1], [1], [0], [0], [0], [0], [1], [0]],
			[[0], [0], [0], [0], [0], [0], [0], [0]],
			[[0], [0], [0], [1], [1], [0], [0], [0]],
			[[0], [0], [0], [1], [1], [0], [0], [0]],
			[[0], [1], [0], [0], [0], [0], [0], [0]],
			[[0], [1], [0], [0], [0], [0], [0], [0]],
			[[0], [0], [0], [0], [0], [0], [0], [0]]
		],

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

	], 
	dtype='float32')

#Index 3 and 5 are our window sizes
input_windowed = tf.reshape(initial_input, [2, 4, 2, 4, 2])

sum_input = tf.reduce_sum(tf.reduce_sum(input_windowed, 4), 2)
print sum_input

box = input_windowed[:, 0, :, 0, :]

window_size = tf.constant([2, 2, 1, 1], dtype='int32')

kernel = tf.pad(tf.ones(window_size), ((1, 0), (1, 0), (0, 0), (0, 0)))

max_window = tf.nn.conv2d(initial_input, kernel, [1, 1, 1, 1], 'SAME')
print max_window

flattened_max = tf.reshape(max_window, [2, -1])
print flattened_max

#batch_last = tf.transpose(max_window, perm = [3, 1, 2, 0])
#print batch_last

max_indeces_flat = tf.argmax(flattened_max, axis=1)

output_list = []
output_list.append(max_indeces_flat // (8))
output_list.append(max_indeces_flat % 8)

max_image_indeces = tf.cast(output_list, dtype='int32')
max_image_windows_padding_top_left = max_image_indeces
max_image_windows_padding_bottom_right = -1 * (max_image_indeces - 7)

max_image_window_padding = tf.stack([max_image_windows_padding_top_left, max_image_windows_padding_bottom_right], axis=1)
print max_image_window_padding

#channel_indeces = tf.expand_dims(tf.range(2), axis=1)
#print channel_indeces

#max_indeces = tf.concat([channel_indeces, max_image_indeces], axis=1)

max_mask = tf.ones([2, 3, 3])



init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run([init_op])

	print initial_input.eval()
	print sum_input.eval()

	#print max_image_windows_padding_top_left.eval()
	#print max_image_windows_padding_bottom_right.eval()
	#print max_image_window_padding.eval()
	#print max_mask.eval()

