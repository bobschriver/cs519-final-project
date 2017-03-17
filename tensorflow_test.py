import tensorflow as tf

with tf.Session() as sess:
	
	initial_input = tf.constant([
	[1, 1, 0, 0, 0, 1, 1, 0],
	[1, 1, 0, 0, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 1, 1, 0, 0, 0],
	[0, 1, 0, 1, 1, 0, 0, 0],
	[0, 1, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0]
	], dtype='float32')

	x_size = tf.constant(8, dtype='int32') 
	y_size = tf.constant(8, dtype='int32')

	x_window_width = tf.constant(2, dtype='int32')
	y_window_width = tf.constant(2, dtype='int32')
	
	split_dis, split_1 = tf.split(initial_input, [1, 7], 0)
	split_dis, split_2 = tf.split(initial_input, [1, 7], 1)
	split_dis, split_3 = tf.split(split_1, [1, 7], 1)
	
	shift_up = tf.pad(split_1, [[0, 1], [0, 0]])
	shift_left = tf.pad(split_2, [[0, 0], [0, 1]])
	shift_up_left = tf.pad(split_3, [[0, 1], [0, 1]])

	max_matrix = tf.add_n([initial_input, shift_up, shift_left, shift_up_left])
	
	'''
	window_size = tf.constant([2, 2], dtype='int32')
	kernel = tf.pad(tf.ones(window_size), ((1, 0), (1, 0), (0, 0), (0, 0)))
	max_matrix = tf.nn.conv2d(initial_input, kernel, [1, 1, 1, 1], 'SAME')
	'''

	init_i = tf.constant(0, dtype='int32')
	init_indeces = tf.TensorArray(dtype=tf.int32, size=4)

	def should_continue(i, max_matrix, indeces):
		return i < 4

	def iteration(i, max_matrix, indeces):
		print max_matrix
		flattened_max = tf.reshape(max_matrix, [-1])
		print flattened_max
		
		max_value = tf.reduce_max(flattened_max)
		#max_index_x = tf.cast(tf.argmax(tf.reduce_max(max_matrix, axis=0), axis=0), dtype='int32')
		#max_index_y = tf.cast(tf.argmax(tf.reduce_max(max_matrix, axis=1), axis=0), dtype='int32')
		flattened_max_index = tf.cast(tf.argmax(flattened_max, axis=0), dtype='int32')
		print flattened_max_index
		max_index_x = tf.cast(flattened_max_index % x_size, dtype='int32')
		max_index_y = tf.cast(flattened_max_index / y_size, dtype='int32')

		# Should change this from 3x3 to based off the window size
		#mask_unpadded = tf.multiply(max_value, tf.ones([3, 3], tf.int32))
		mask_unpadded = tf.fill([2, 2], max_value)
		padding_x = tf.stack([max_index_x, x_size - (max_index_x + x_window_width)])
		padding_y = tf.stack([max_index_y, y_size - (max_index_y + y_window_width)])
		paddings = tf.stack([padding_y, padding_x])
		
		mask = tf.pad(mask_unpadded, paddings)
		
		indeces = indeces.write(i, [max_index_y, max_index_x])

		return i + 1, max_matrix - mask, indeces

	i, max_masked, indeces = tf.while_loop(should_continue, iteration, [init_i, max_matrix, init_indeces])
	
	indeces = indeces.stack()

	manhattan_distance = tf.reshape(tf.reduce_sum(indeces, axis=1), [4, 1])
	
	#Do padding here to support loop shape invariance
	def continue_populate(x, *args):
		return x < 2

	def populate_column(i, row_output, manhattan_distance, indeces, initial_input):
		min_manhattan_distance = tf.cast(tf.argmin(manhattan_distance, axis=0), dtype=tf.int32)

		top_left_index = tf.gather(indeces, min_manhattan_distance)

		neighborhood_indeces = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])
		neighborhood_indeces = tf.add(neighborhood_indeces, top_left_index)

		neighborhood_values = tf.reshape(tf.gather_nd(initial_input, neighborhood_indeces), [2, 2])
	
		padding_x = tf.stack([i * 2, ((x_size / 4) - (i + 1)) * 2])
		padding_y = tf.constant([0, 0])
		neighborhood_values_padded = tf.pad(neighborhood_values, [padding_y, padding_x])

		min_index_scalar = tf.squeeze(min_manhattan_distance)
		m_padding_x = tf.stack([min_index_scalar, 4 - (min_index_scalar + 1)])
		m_padding_y = tf.constant([0, 0])

		m_update = tf.reshape(tf.constant(100, dtype=tf.int32), [1, 1])
		m_update_padded = tf.pad(m_update, [m_padding_x, m_padding_y])
		m_update_padded.set_shape([4, 1])

		return [i + 1, row_output + neighborhood_values_padded, manhattan_distance + m_update_padded, indeces, initial_input]

	def populate_row(j, output, manhattan_distance, indeces, initial_input):
		init_i = tf.constant(0, dtype=tf.int32)
		init_row = tf.zeros([2, 4], dtype=tf.float32)

		i, row_output, manhattan_distance, indeces, initial_input = tf.while_loop(continue_populate, populate_column, [init_i, init_row, manhattan_distance, indeces, initial_input])
		
		padding_x = tf.constant([0, 0])
		padding_y = tf.stack([j * 2, ((y_size / 4) - (j + 1)) * 2])
		row_padded = tf.pad(row_output, [padding_y, padding_x])

		return [j + 1, output + row_padded, manhattan_distance, indeces, initial_input]
	
	init_j = tf.constant(0, dtype=tf.int32)
	init_output = tf.zeros([4, 4], dtype=tf.float32)

	j, output, manhattan_distance, indeces, initial_input = tf.while_loop(continue_populate, populate_row, [init_j, init_output, manhattan_distance, indeces, initial_input])
	
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	print sess.run([output, manhattan_distance, indeces, initial_input, max_matrix])

writer.close()
