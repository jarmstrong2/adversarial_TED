import tensorflow as tf
import numpy as np

class RNN_t_model(object):

	def __init__(self, is_training, config):
		batch_size = config.batch_size
		vocal_len = config.vocal_len
		hidden_size = config.hidden_size_RNN_t
		output_size = config.output_size_RNN_t

		self.cu = tf.placeholder(tf.float32, [batch_size, None, vocal_len])

		# maybe change bias?!
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
		if is_training and config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    			lstm_cell, output_keep_prob=config.keep_prob
    		)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers_RNN_t, state_is_tuple=True)
		seq_len = self.cu_seq_len(self.cu)

		output, _ = tf.nn.dynamic_rnn(
      		cell,
      		self.cu,
      		dtype=tf.float32,
      		sequence_length=seq_len,
      	)

		relevant_outputs = self.last_relevant(output, tf.identity(seq_len))
		
		output_w = tf.get_variable("RNN_t_w", [hidden_size, output_size])
		output_b = tf.get_variable("RNN_t_b", [output_size])
        
		self.output=seq_len
		self.output = tf.matmul(relevant_outputs, output_w) + output_b

	def cu_seq_len(self, data):
		''' Assuming one-hot char matrix is batchsize x max speech length x vocab length, return
		sequence length for each char matrix '''

		length = tf.reduce_sum(tf.reduce_sum(data, reduction_indices=2), reduction_indices=1)
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
		output_size_RNN_t = 40
		num_layers_RNN_t = 3
		keep_prob = 0.5

	with tf.Graph().as_default(), tf.Session() as session:
		initializer = tf.random_uniform_initializer(-0.4,0.4)
		mod = RNN_t_model(True, configobj())
		tf.initialize_all_variables().run()
		u = np.eye(50,30)
		u = np.expand_dims(u,0)
		r = np.concatenate((u,u,u),axis=0)
		result = session.run(tf.shape(mod.output), {mod.cu:r})
		print(result)