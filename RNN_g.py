from RNN_t import *
import tensorflow as tf
import numpy as np

class RNN_g_model(object):

	def __init__(self, is_training, config, RNN_t_output):
		batch_size = config.batch_size
		straight_size = config.straight_size
		max_time_steps = config.max_time_steps
		hidden_size = config.hidden_size_RNN_g
		z_size = config.z_size
		output_size_RNN_t = config.output_size_RNN_t

		self.z = tf.placeholder(tf.float32, [batch_size, z_size])

		# linear trans for z -> straight vector
		f_w = tf.get_variable("RNN_g_w", [z_size, straight_size])
		f_b = tf.get_variable("RNN_g_b", [straight_size])

		init_straight = tf.matmul(self.z, f_w) + f_b

		# linear trans for straight vector -> hidden_size of lstm
		h_w = tf.get_variable("RNN_h_w", [straight_size + output_size_RNN_t, hidden_size])
		h_b = tf.get_variable("RNN_h_b", [hidden_size])

		# linear trans for hidden_size of lstm -> straight vector
		j_w = tf.get_variable("RNN_j_w", [hidden_size, straight_size])
		j_b = tf.get_variable("RNN_j_b", [straight_size])

		# maybe change bias?!
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
		if is_training and config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    			lstm_cell, output_keep_prob=config.keep_prob
    		)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers_RNN_g, state_is_tuple=True)

		outputs = []
		concat_straight_RNN_t = tf.concat(1, [init_straight, RNN_t_output])
		if is_training and config.keep_prob < 1:
			concat_straight_RNN_t = tf.nn.dropout(concat_straight_RNN_t, config.keep_prob)
		cell_output = tf.matmul(concat_straight_RNN_t, h_w) + h_b
		state = cell.zero_state(batch_size, tf.float32)

		# constructs the RNN_g
		# at time t :
		#	input : [(1) straight vector, (2) output from RNN_t] -> [vector of size hidden_size]
		#	output : [vector of size hidden_size] -> [straight vector]
		# at time t+1:
		#	input : [(1) output time t, (2) output from RNN_t] -> [vector of size hidden_size]
		with tf.variable_scope("RNN"):
			for time_step in range(max_time_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(cell_output, state)
				straight = tf.matmul(cell_output, j_w) + j_b
				outputs.append(straight)
				concat_straight_RNN_t = tf.concat(1, [straight, RNN_t_output])
				if is_training and config.keep_prob < 1:
					concat_straight_RNN_t = tf.nn.dropout(concat_straight_RNN_t, config.keep_prob)
				cell_output = tf.matmul(concat_straight_RNN_t, h_w) + h_b
				
		self.outputs = outputs

# testing if everything works
if __name__ == "__main__" :
	class configobj(object):
		batch_size = 3
		vocal_len = 30
		hidden_size_RNN_t = 10
		output_size_RNN_t = 40
		num_layers_RNN_t = 3
		keep_prob = 0.5
		straight_size = 50
		max_time_steps = 10
		hidden_size_RNN_g = 200
		num_layers_RNN_g = 3
		z_size = 10

	with tf.Graph().as_default(), tf.Session() as session:
		initializer = tf.random_uniform_initializer(-0.4,0.4)
		with tf.variable_scope("mod_t", reuse=None, initializer=initializer):
			mod_t = RNN_t_model(True, configobj())
		with tf.variable_scope("mod_g", reuse=None, initializer=initializer):
			mod_g = RNN_g_model(True, configobj(), mod_t.outputs)
		tf.initialize_all_variables().run()
		u = np.eye(50,30)
		u = np.expand_dims(u,0)
		r = np.concatenate((u,u,u),axis=0)
		z = np.random.randn(3, 10)
		result = session.run((mod_g.outputs), {mod_t.cu:r, mod_g.z:z})
		print(result)
