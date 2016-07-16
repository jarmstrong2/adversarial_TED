import tensorflow as tf

# windowing layer
#
# input_data	- input transformed by linear nn using input from LSTM below
#				- size: [batchsize x mixture size * 3]
# cu			- one-hot vector matrix describing speech text
#				- size: [batchsize x speech length x vocab length]
# kappas_t_1	- previous kappa from t - 1 windowing calculations
#				- size: [batchsize x mixture size]

def w(input_data, cu, kappas_t_1, config):
	
	batch_size = config.batch_size
	mixture_size = config.mixture_size
	vocab_length = config.vocab_length

	# split along dim of mixture size * 3
	hat_alphas_t, hat_betas_t, hat_kappas_t = tf.split(1, 3, input_data)

	alphas_t = tf.exp(hat_alphas_t)
	betas_t = tf.exp(hat_betas_t)
	kappas_t = tf.add(kappas_t_1, tf.exp(hat_kappas_t))

	speech_length = tf.shape(cu)[1]

	u = tf.linspace(1.0, tf.cast(speech_length,tf.float32) , speech_length)
	u = tf.expand_dims(u, 0)
	u = tf.expand_dims(u, 0)
	u = tf.tile(u, [batch_size, mixture_size, 1])

	alphas_t_expanded = tf.tile(tf.expand_dims(alphas_t, -1), [1, 1, speech_length])
	betas_t_expanded = tf.tile(tf.expand_dims(betas_t, -1), [1, 1, speech_length])
	kappas_t_expanded = tf.tile(tf.expand_dims(kappas_t, -1), [1, 1, speech_length])

	calc = tf.square(tf.sub(kappas_t_expanded, u))
	calc = tf.mul(calc, tf.neg(betas_t_expanded))
	calc = tf.exp(calc)
	calc = tf.mul(calc, alphas_t_expanded)

	phi_t = tf.expand_dims(tf.reduce_sum(calc, 1), 1)

	output = tf.squeeze(tf.batch_matmul(phi_t, cu), [1])

	return output, kappas_t, phi_t
