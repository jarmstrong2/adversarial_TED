import tensorflow as tf
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class RNN_MNIST_model(object):
	def __init__(self, config, is_training = True, model_type="FULL"):
		batch_size = config.batch_size
		z_size = config.z_size

		lstm_layers_RNN_g = config.lstm_layers_RNN_g
		lstm_layers_RNN_d = config.lstm_layers_RNN_d

		hidden_size_RNN_g = config.hidden_size_RNN_g
		hidden_size_RNN_d = config.hidden_size_RNN_d


		self.target = tf.placeholder(tf.float32, [batch_size, 10])

		self.target_bin = tf.placeholder(tf.float32, [batch_size, 2])

		self.trainables_variables = []

		# --------------RNN_g----------------

		if model_type == "GEN" or model_type == "FULL":
			self.z = tf.placeholder(tf.float32, [batch_size, z_size])

			# linear trans for z -> hidden_size_RNN_g
			f_w = tf.get_variable("RNN_g_w", [z_size, hidden_size_RNN_g])
			f_b = tf.get_variable("RNN_g_b", [hidden_size_RNN_g])

			self.trainables_variables.append(f_w)
			self.trainables_variables.append(f_b)

			init_state = tf.matmul(self.z, f_w) + f_b
			collected_state = ((init_state, init_state),)
			for layer in range(config.lstm_layers_RNN_g - 1):
				collected_state += ((init_state, init_state),)

			init_image = tf.zeros([batch_size,28*28])

			init_input = tf.concat(1, [init_image, self.target])

			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_RNN_g, forget_bias=0.0, state_is_tuple=True)
			if is_training and config.keep_prob < 1:
				lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
	    			lstm_cell, output_keep_prob=config.keep_prob
	    		)
			cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.lstm_layers_RNN_g, state_is_tuple=True)

			# linear trans for [x_image_size * y_image_size + num_classes] -> hidden_size_RNN_g
			g_w = tf.get_variable("RNN_g_input_target_w", [(28*28)+10, hidden_size_RNN_g])
			g_b = tf.get_variable("RNN_g_input_target_b", [hidden_size_RNN_g])

			self.trainables_variables.append(g_w)
			self.trainables_variables.append(g_b)

			# linear trans for hidden_size_RNN_g -> [x_image_size * y_image_size]
			h_w = tf.get_variable("RNN_g_output_target_w", [hidden_size_RNN_g, (28*28)])
			h_b = tf.get_variable("RNN_g_output_target_b", [(28*28)])

			self.trainables_variables.append(h_w)
			self.trainables_variables.append(h_b)

			output = []
			if is_training:
				init_input = tf.nn.dropout(init_input, config.keep_prob)
			cell_input = tf.matmul(init_input, g_w) + g_b
			self.state = state = collected_state

			lstm_variables = []

			with tf.variable_scope("RNN_g") as vs:
				for time_step in range(1):
					if time_step > 0: tf.get_variable_scope().reuse_variables()
					(cell_output, state) = cell(tf.nn.relu(cell_input), state)
					cell_output = tf.matmul(cell_output, h_w) + h_b
					output.append(cell_output)
					new_input = tf.concat(1, [cell_output, self.target])
					if is_training:
						new_input = tf.nn.dropout(new_input, config.keep_prob)
					cell_input = tf.matmul(new_input, g_w) + g_b

				lstm_variables = [v for v in tf.all_variables()
                    if v.name.startswith(vs.name)]

			self.trainables_variables += lstm_variables

			outputs_RNN_g = tf.transpose(output, perm=[1,0,2])
			outputs_RNN_g = tf.sigmoid(outputs_RNN_g)
			#outputs_RNN_g = tf.round(outputs_RNN_g)

			if model_type == "GEN":	
				self.outputs = outputs_RNN_g

		# ------------------------------------

		# --------------RNN_d----------------

		if model_type == "DISC" or model_type == "FULL":
			if model_type == "DISC":
				self.image_input = tf.placeholder(tf.float32, [batch_size, 1, 28*28])
			else:
				self.image_input = outputs_RNN_g

			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_RNN_d, forget_bias=0.0, state_is_tuple=True)
			if is_training and model_type == "DISC" and config.keep_prob < 1:
				lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
	    			lstm_cell, output_keep_prob=config.keep_prob
	    		)
			cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.lstm_layers_RNN_d, state_is_tuple=True)

			# linear trans for [x_image_size * y_image_size + num_classes] -> hidden_size_RNN_g
			i_w = tf.get_variable("RNN_i_target_w", [10, hidden_size_RNN_d])
			i_b = tf.get_variable("RNN_i_target_b", [hidden_size_RNN_d])

			if model_type == "DISC":
				self.trainables_variables.append(i_w)
				self.trainables_variables.append(i_b)

			init_state_input = tf.matmul(self.target, i_w) + i_b

			init_state = ((init_state_input,init_state_input),)
			for layer in range(config.lstm_layers_RNN_d - 1):
				init_state += ((init_state_input,init_state_input),)

			lstm_variables = []

			with tf.variable_scope("RNN_d") as vs:
				output, _ = tf.nn.dynamic_rnn(
		      		cell,
		      		self.image_input,
		      		initial_state = init_state,
		      		dtype=tf.float32,
		      	)

				lstm_variables = [v for v in tf.all_variables()
                    if v.name.startswith(vs.name)]		
            
			if model_type == "DISC":
				self.trainables_variables += lstm_variables

			# linear trans for hidden_size of lstm -> single value
			j_w = tf.get_variable("RNN_j_prob_w", [hidden_size_RNN_d, 2])
			j_b = tf.get_variable("RNN_j_prob_b", [2])

			if model_type == "DISC":
				self.trainables_variables.append(j_w)
				self.trainables_variables.append(j_b)

			final_output = tf.squeeze(output, [1])
			final_trans = tf.matmul(final_output, j_w) + j_b
			y = tf.nn.softmax(final_trans)

			cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target_bin * tf.log(y), reduction_indices=[1]))

			self.cost = cross_entropy

			correct_pred = tf.equal(tf.argmax(final_trans,1), tf.argmax(self.target_bin,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

			self.lr = config.lr
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.trainables_variables),
			                                   config.max_grad_norm)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			self.train_op = self.optimizer.apply_gradients(zip(grads, self.trainables_variables))

		# ------------------------------------

def getinput(batch_x):
	# spliting the image into its quadrants, then flattening, then concatenating
	full_im = np.reshape(batch_x,(configobj().batch_size/2, 28,28))
	input_x = np.reshape(full_im, (configobj().batch_size/2, 28*28))
	input_x = np.expand_dims(input_x, axis=1)
	return input_x

if __name__ == "__main__" :
	class configobj(object):
		batch_size = 2**7
		keep_prob = 0.5
		z_size = 100
		lstm_layers_RNN_g = 8
		lstm_layers_RNN_d = 8
		hidden_size_RNN_g = 600
		hidden_size_RNN_d = 600
		#lr = 0.005
		lr = 0.001
		max_grad_norm = 5
		iterations = (10**6)
		init_scale = 0.01

	class configobj_g(object):
		batch_size = 2**6
		keep_prob = 0.5
		z_size = 100
		lstm_layers_RNN_g = 8
		lstm_layers_RNN_d = 8
		hidden_size_RNN_g = 600
		hidden_size_RNN_d = 600
		#lr = 0.005
		lr = 0.001
		max_grad_norm = 5
		iterations = 10**5
		init_scale = 0.01

	with tf.Graph().as_default(), tf.Session() as session:
		initializer = tf.random_uniform_initializer(-configobj().init_scale,configobj().init_scale)

		with tf.variable_scope("model_full", reuse=None, initializer=initializer):
			mod_f = RNN_MNIST_model(configobj(), True, model_type="FULL")
		with tf.variable_scope("model_full", reuse=True, initializer=initializer):
			mod_g = RNN_MNIST_model(configobj_g(), False, model_type="GEN")
		with tf.variable_scope("model_full", reuse=True, initializer=initializer):
			mod_d = RNN_MNIST_model(configobj(), True, model_type="DISC")

		tf.initialize_all_variables().run()
		saver = tf.train.Saver()
		x_plot_class = []
		y_plot_class = []
		accumulator_class = 0
		stepsingen_class = 0

		x_plot_loss= []
		y_plot_loss = []
		accumulator_loss = 0
		stepsingen_loss = 0

		for i in range(configobj().iterations):
			if ((i+1) % 1000 == 0):
				print("------------")
				print("Step: {}".format(i+1))
				
				print("************")
				#print(cost_gen_g)
				print("Loss: {}, Accuracy: {}".format(cost_gen_g, acc_gen_g))
				print("************")

				#print((cost + cost_gen) / 2)
				print("Loss: {}, Accuracy: {}".format(cost, acc))

				x_plot_class.append(i)
				y_plot_class.append(accumulator_class/stepsingen_class)

				accumulator_class = 0
				stepsingen_class = 0

				plt.figure()
				plt.plot(x_plot_class, y_plot_class, 'ro')
				plt.savefig('classification.png')

				x_plot_loss.append(i)
				y_plot_loss.append(accumulator_loss/stepsingen_loss)

				accumulator_loss = 0
				stepsingen_loss = 0

				plt.figure()
				plt.plot(x_plot_loss, y_plot_loss, 'ro')
				plt.savefig('loss.png')


			# update the generator
			if ((i+1) % 3 == 0):
				z = np.random.uniform(-1,1,(configobj().batch_size,configobj().z_size))

				# randomly generating one-hot vect to describe gen number image segments
				target_gen = np.zeros((configobj().batch_size, 10))
				ind = [np.random.choice(10) for row in target_gen]
				target_gen[range(target_gen.shape[0]), ind] = 1
				target_gen_bin = np.zeros((configobj().batch_size, 2))
				target_gen_bin[:,0] = 1

				_, cost_gen_g, acc_gen_g = session.run((mod_f.train_op, mod_f.cost, mod_f.accuracy), {mod_f.z:z, mod_f.target_bin:target_gen_bin, mod_f.target:target_gen})
				
				accumulator_class += acc_gen_g
				stepsingen_class += 1

				accumulator_loss += cost_gen_g
				stepsingen_loss += 1
			# update the discriminator
			else :
				batch_x, batch_y = mnist.train.next_batch(configobj().batch_size/2)
				batch_x = getinput(batch_x)

				target_bin = np.zeros((configobj().batch_size/2, 2))
				target_bin[:,0] = 1

				z = np.random.uniform(-1,1,(configobj().batch_size/2,configobj().z_size))

				# randomly generating one-hot vect to describe gen number image segments
				target_gen = np.zeros((configobj().batch_size/2, 10))
				ind = [np.random.choice(10) for row in target_gen]
				target_gen[range(target_gen.shape[0]), ind] = 1
				target_gen_bin = np.zeros((configobj().batch_size/2, 2))
				target_gen_bin[:,1] = 1

				gen_x = session.run((mod_g.outputs), {mod_g.z:z, mod_g.target:target_gen, mod_g.target_bin:target_gen_bin})
				print(gen_x);
				# trying to shuffle fake and real data
				x = np.concatenate((batch_x, gen_x), axis=0)
				t = np.concatenate((batch_y, target_gen), axis=0)
				y = np.concatenate((target_bin, target_gen_bin), axis=0)

				c = np.concatenate((x.reshape(len(x), -1), y.reshape(len(y), -1), t.reshape(len(t), -1)), axis=1)
				
				np.random.shuffle(c)

				x = c[:, :x.size//len(x)].reshape(x.shape)
				y = c[:, x.size//len(x):(x.size//len(x))+(y.size//len(y))].reshape(y.shape)
				t = c[:, (x.size//len(x))+(y.size//len(y)):].reshape(t.shape)

				_, cost, acc = session.run((mod_d.train_op, mod_d.cost, mod_d.accuracy), {mod_d.target_bin:y, mod_d.target:t, mod_d.image_input:x})

			if ((i+1) % 100000 == 0):
				save_path = saver.save(session, "model.ckpt")
				print("Model saved in file: %s" % save_path)







