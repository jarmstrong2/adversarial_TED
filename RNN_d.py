from RNN_t import *
from RNN_g import *
import tensorflow as tf
import numpy as np

class RNN_d_model(object):

	def __init__(self, is_training, config, RNN_g_output, RNN_t_output, model_type="FULL"):
		batch_size = config.batch_size
		straight_size = config.straight_size
		max_time_steps = config.max_time_steps
		hidden_size = config.hidden_size_RNN_d
		output_size_RNN_t = config.output_size_RNN_t

		if model_type == "FULL":
			# Full generative adverserial network
			self.data = RNN_g_output
		else:
			# Using only the discriminative RNN
			self.data = tf.placeholder(tf.float32, [batch_size, max_time_steps, straight_size])

		# maybe change bias?!
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
		if is_training and config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    			lstm_cell, output_keep_prob=config.keep_prob
    		)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers_RNN_d, state_is_tuple=True)
		seq_len = self.speech_data_seq_len(self.data)

		# concat with RNN_t outputs
		RNN_t_output = tf.expand_dims(RNN_t_output, 1)
		RNN_t_output = tf.tile(RNN_t_output, [1,max_time_steps,1])
		data_concat = tf.concat(2, [self.data, RNN_t_output])

		if is_training and config.keep_prob < 1:
			data_concat = tf.nn.dropout(data_concat, config.keep_prob)

		inputs = data_concat

		output, _ = tf.nn.dynamic_rnn(
      		cell,
      		inputs,
      		dtype=tf.float32,
      		sequence_length=seq_len,
      		)

		# linear trans for hidden_size of lstm -> single value
		k_w = tf.get_variable("RNN_k_w", [hidden_size, 1])
		k_b = tf.get_variable("RNN_k_b", [1])

		# output at end of each sequence
		final_output = self.last_relevant(output, seq_len)
		final_trans = tf.matmul(final_output, k_w) + k_b
		final_prob = tf.sigmoid(final_trans)

		self.outputs = final_prob

	def speech_data_seq_len(self, data):
		''' Assuming one-hot char matrix is batchsize x max speech length x vocab length, return
		sequence length for each char matrix '''

		signed_data = tf.sign(tf.reduce_sum(tf.abs(data), reduction_indices=2))
		length = tf.reduce_sum(signed_data, reduction_indices=1)
		return length

	def last_relevant(self, output, length):
		length = tf.cast(length, tf.int32)
		batch_size = tf.shape(output)[0]
		max_length = tf.shape(output)[1]
		output_size = tf.shape(output)[2]
		index = tf.range(0, batch_size) * max_length + (length - 1)
		flat = tf.reshape(output, [-1, output_size])
		relevant = tf.gather(flat, index)
		return relevant

# testing if everything works
if __name__ == "__main__" :
	class configobj(object):
		batch_size = 3
		vocal_len = 30
		hidden_size_RNN_t = 10
		hidden_size_RNN_d = 10
		output_size_RNN_t = 40
		num_layers_RNN_t = 3
		num_layers_RNN_d = 3
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
		with tf.variable_scope("mod_d", reuse=None, initializer=initializer):
			mod_d = RNN_d_model(True, configobj(), mod_g.outputs, mod_t.outputs)
		tf.initialize_all_variables().run()
		u = np.eye(50,30)
		u = np.expand_dims(u,0)
		r = np.concatenate((u,u,u),axis=0)
		z = np.random.randn(3, 10)
		result = session.run((tf.shape(mod_d.outputs)), {mod_t.cu:r, mod_g.z:z})
		print(result)
