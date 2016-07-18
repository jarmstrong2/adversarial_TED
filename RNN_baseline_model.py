import tensorflow as tf
import numpy as np

class RNN_baseline_model(object):

	def __init__(self, is_training, config, model_type="FULL"):
		batch_size = config.batch_size
		straight_size = config.straight_size
		max_time_steps = config.max_time_steps
		vocal_len = config.vocal_len

		hidden_size_RNN_t = config.hidden_size_RNN_t
		hidden_size_RNN_g = config.hidden_size_RNN_g
		hidden_size_RNN_d = config.hidden_size_RNN_d
		
		output_size_RNN_t = config.output_size_RNN_t

		z_size = config.z_size

		# --------------RNN_t----------------

		self.cu = tf.placeholder(tf.float32, [batch_size, None, vocal_len])

		lstm_cell_RNN_T = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_RNN_t, forget_bias=0.0, state_is_tuple=True)
		if is_training and config.keep_prob < 1:
			lstm_cell_RNN_T = tf.nn.rnn_cell.DropoutWrapper(
    			lstm_cell_RNN_T, output_keep_prob=config.keep_prob
    		)
		cell_RNN_T = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_RNN_T] * config.num_layers_RNN_t, state_is_tuple=True)
		seq_len_RNN_t = self.cu_seq_len(self.cu)

		with tf.variable_scope("RNN_t") as scope:
			output_RNN_t, _ = tf.nn.dynamic_rnn(
	      		cell_RNN_T,
	      		self.cu,
	      		dtype=tf.float32,
	      		sequence_length=seq_len_RNN_t,
	      	)

			relevant_outputs_RNN_t = self.last_relevant(output_RNN_t, tf.identity(seq_len_RNN_t))
			
			output_w = tf.get_variable("RNN_t_w", [hidden_size_RNN_t, output_size_RNN_t])
			output_b = tf.get_variable("RNN_t_b", [output_size_RNN_t])
	        
			outputs_RNN_t = tf.matmul(relevant_outputs_RNN_t, output_w) + output_b

			scope.reuse_variables()

			outputs_RNN_t_reuse = outputs_RNN_t	

		# -----------------------------------

		# --------------RNN_g----------------

		if model_type == "GEN" or model_type == "FULL":
			self.z = tf.placeholder(tf.float32, [batch_size, z_size])

			# linear trans for z -> straight vector
			f_w = tf.get_variable("RNN_g_w", [z_size, straight_size])
			f_b = tf.get_variable("RNN_g_b", [straight_size])

			init_straight = tf.matmul(self.z, f_w) + f_b

			# linear trans for straight vector -> hidden_size of lstm
			h_w = tf.get_variable("RNN_h_w", [straight_size + output_size_RNN_t, hidden_size_RNN_g])
			h_b = tf.get_variable("RNN_h_b", [hidden_size_RNN_g])

			# linear trans for hidden_size of lstm -> straight vector
			j_w = tf.get_variable("RNN_j_w", [hidden_size_RNN_g, straight_size])
			j_b = tf.get_variable("RNN_j_b", [straight_size])

			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_RNN_g, forget_bias=0.0, state_is_tuple=True)
			if is_training and config.keep_prob < 1:
				lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
	    			lstm_cell, output_keep_prob=config.keep_prob
	    		)
			cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers_RNN_g, state_is_tuple=True)

			outputs_RNN_g = []
			concat_straight_RNN_t = tf.concat(1, [init_straight, outputs_RNN_t])
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
			with tf.variable_scope("RNN_g"):
				for time_step in range(max_time_steps):
					if time_step > 0: tf.get_variable_scope().reuse_variables()
					(cell_output, state) = cell(cell_output, state)
					straight = tf.matmul(cell_output, j_w) + j_b
					outputs_RNN_g.append(straight)
					concat_straight_RNN_t = tf.concat(1, [straight, outputs_RNN_t_reuse])
					if is_training and config.keep_prob < 1:
						concat_straight_RNN_t = tf.nn.dropout(concat_straight_RNN_t, config.keep_prob)
					cell_output = tf.matmul(concat_straight_RNN_t, h_w) + h_b
			
			outputs_RNN_g = tf.transpose(outputs_RNN_g, perm=[1,0,2])

			if model_type == "GEN":	
				self.outputs = outputs_RNN_g

		# -----------------------------------

		# --------------RNN_d----------------

		if model_type == "DISC" or model_type == "FULL":
			if model_type == "FULL":
				# Full generative adverserial network
				self.data = outputs_RNN_g
			elif model_type == "DISC":
				# Using only the discriminative RNN
				self.data = tf.placeholder(tf.float32, [batch_size, max_time_steps, straight_size])

			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_RNN_d, forget_bias=0.0, state_is_tuple=True)
			if is_training and config.keep_prob < 1:
				lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
	    			lstm_cell, output_keep_prob=config.keep_prob
	    		)
			cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers_RNN_d, state_is_tuple=True)
			seq_len = self.speech_data_seq_len(self.data)

			# concat with RNN_t outputs
			outputs_RNN_t = tf.expand_dims(outputs_RNN_t_reuse, 1)
			outputs_RNN_t = tf.tile(outputs_RNN_t, [1,max_time_steps,1])
			data_concat = tf.concat(2, [self.data, outputs_RNN_t])

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
			k_w = tf.get_variable("RNN_k_w", [hidden_size_RNN_d, 1])
			k_b = tf.get_variable("RNN_k_b", [1])

			# output at end of each sequence
			final_output = self.last_relevant(output, seq_len)
			final_trans = tf.matmul(final_output, k_w) + k_b
			final_prob = tf.sigmoid(final_trans)

			self.outputs = final_prob
			self.cost = -tf.log(final_prob)
			self.cost = tf.squeeze(self.cost, [1])

			# TODO figure out learning procedure
			# self._lr = config.lr
			# tvars = tf.trainable_variables()
			# grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
			#                                    config.max_grad_norm)
			# optimizer = tf.train.AdamOptimizer(self.lr)
			# self._train_op = optimizer.apply_gradients(zip(grads, tvars))

		# -----------------------------------

	def cu_seq_len(self, data):
		''' Assuming one-hot char matrix is batchsize x max speech length x vocab length, return
		sequence length for each char matrix '''

		length = tf.reduce_sum(tf.reduce_sum(data, reduction_indices=2), reduction_indices=1)
		return length

	def speech_data_seq_len(self, data):
		''' Assuming one-hot char matrix is batchsize x max speech length x vocab length, return
		sequence length for each char matrix '''

		signed_data = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
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
		with tf.variable_scope("model_full", reuse=None, initializer=initializer):
			mod = RNN_baseline_model(False, configobj())
		with tf.variable_scope("model_full", reuse=True, initializer=initializer):
			mod_g = RNN_baseline_model(False, configobj(), model_type="GEN")
		with tf.variable_scope("model_full", reuse=True, initializer=initializer):
			mod_d = RNN_baseline_model(False, configobj(), model_type="DISC")
		tf.initialize_all_variables().run()
		u = np.eye(50,30)
		u = np.expand_dims(u,0)
		r = np.concatenate((u,u,u),axis=0)
		z = np.random.randn(3, 10)
		g = np.random.randn(3, 10, 50)
		result = session.run((tf.shape(mod.outputs)), {mod.cu:r, mod.z:z})
		print(result)
		result = session.run((tf.shape(mod_g.outputs)), {mod_g.cu:r, mod_g.z:z})
		print(result)
		result = session.run((tf.shape(mod_d.outputs)), {mod_d.cu:r, mod_d.data:g})
		print(result)
