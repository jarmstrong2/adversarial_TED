import tensorflow as tf
import numpy as np

class RNN_g_model(object):

	def __init__(self, is_training, config, RNN_t_output):
		batch_size = config.batch_size
		straight_size = config.straight_size
		max_time_steps = config.max_time_steps
		hidden_size = config.hidden_size_RNN_g
		layer_size = config.layer_size_RNN_g
		z_size = config.z_size
		output_size_RNN_t = config.output_size_RNN_t

		z = tf.placeholder(tf.float32, [batch_size, z_size])

		# linear trans for z -> straight vector
		f_w = tf.get_variable("RNN_g_w", [z_size, straight_size])
		f_b = tf.get_variable("RNN_g_b", [straight_size])

		init_straight = tf.matmul(z, f_w) + f_b

		# linear trans for straight vector -> hidden_size of lstm
		h_w = tf.get_variable("RNN_h_w", [straight_size + output_size_RNN_t, hidden_size])
		h_b = tf.get_variable("RNN_h_b", [hidden_size])

		# linear trans for hidden_size of lstm -> straight vector
		j_w = tf.get_variable("RNN_j_w", [straight_size, hidden_size])
		j_b = tf.get_variable("RNN_j_b", [hidden_size])

		# maybe change bias?!
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
		if is_training and config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    			lstm_cell, output_keep_prob=config.keep_prob
    		)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers_RNN_g, state_is_tuple=True)

		outputs = []
		cell_output = tf.matmul(tf.concat(1, [init_straight, RNN_t_output]), h_w) + h_b
		state = cell.zero_state(batch_size, tf.float32)

		# constructs the RNN_g
		# at time t :
		#	input : [(1) straight vector, (2) output from RNN_t] -> [vector of size hidden_size]
		#	output : [vector of size hidden_size] -> [straight vector]
		# at time t+1:
		#	input : [(1) output time t, (2) output from RNN_t] -> [vector of size hidden_size]
		with tf.variable_scope("RNN"):
			for time_step in range(max_time_steps):
				tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(cell_output, state)
				straight = tf.matmul(cell_output, j_w) + j_b
				cell_output = tf.matmul(tf.concat(1, [straight, RNN_t_output]), h_w) + h_b
				outputs.append(cell_output)

		self.outputs = outputs

		













