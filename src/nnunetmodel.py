import tensorflow as tf
import config as cf

IN_CHANNELS = cf.nnunet_IN_CHANNELS 
BASE_FILTER = cf.nnunet_BASE_FILTER
FIRST_OUT_CHANNEL = cf.nnunet_FIRST_OUT_CHANNEL
CONV_STRIDE = cf.nnunet_CONV_STRIDE
leaky_relu_alpha = cf.nnunet_leaky_relu_alpha
OUT_LABELs = cf.nnunet_OUT_LABELs
DTYPE = cf.nnunet_DTYPE
no_of_encoding_layers = cf.nnunet_NO_OF_ENCODING_LAYERS

def weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=cf.nnunet_initialization_std))


def bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(cf.nnunet_initialization_std, dtype=DTYPE))


def convInLRelu(inlayer,n_in_maps, n_out_maps):
	"""
	"""
	#====Conv operation=====
	W = weight_variable(name='weight', shape=BASE_FILTER+[n_in_maps,n_out_maps])
	b = bias_variable(name='bias', shape=[n_out_maps])
	conv = tf.nn.conv3d(inlayer,filter=W,strides=CONV_STRIDE, padding='SAME', name='conv')
	conv = conv + b
	convIn = tf.contrib.layers.instance_norm(conv)
	convInLeakyrelu = tf.nn.leaky_relu(convIn, leaky_relu_alpha)

	return convInLeakyrelu

def down_sample(inlayer):

	maxpool = tf.nn.max_pool3d(inlayer,ksize=[1,2,2,2,1],strides=[1,2,2,2,1], padding= 'VALID')

	return maxpool

def upsample(inlayer):
	"""
	Up samples 3d volume to twice its size and halves the number of channels
	"""
	t_upsampling = tf.keras.layers.UpSampling3D(size=(2,2,2), data_format="channels_last")(inlayer)

	return t_upsampling

def final_layer(inlayer, n_in_maps, n_out_maps):

	W = weight_variable(name='final_weight', shape=BASE_FILTER+[n_in_maps,n_out_maps])
	b = bias_variable(name='final_bias', shape=[n_out_maps])
	conv = tf.nn.conv3d(inlayer,filter=W,strides=CONV_STRIDE, padding='SAME', name='final_conv')
	conv = conv + b
	softmax = tf.nn.softmax(conv)

	return softmax

def nnunet(input_image, n_encoding = no_of_encoding_layers,output_check=False):

	level = 0
	inlayer = input_image
	n_out_maps = FIRST_OUT_CHANNEL
	level_out = {}
	output_shape=[]
	for level in range(n_encoding):
		with tf.variable_scope('enc_level{}b1'.format(level)):
			block1 = convInLRelu(inlayer, n_in_maps=inlayer.shape.as_list()[4], n_out_maps= n_out_maps)
			output_shape.append(block1.shape.as_list())
		with tf.variable_scope('enc_level{}b2'.format(level)):
			block2 = convInLRelu(block1, n_in_maps=block1.shape.as_list()[4], n_out_maps= n_out_maps)
			output_shape.append(block2.shape.as_list())
		encoded = down_sample(block2)
		output_shape.append(encoded.shape.as_list())
		level_out[level] = block2
		
		n_out_maps = 2*n_out_maps
		inlayer = encoded

	#===Decoding====
	#n_out_maps = 2*n_out_maps

	for level in range(n_encoding): 
		with tf.variable_scope('dec_level{}b1'.format(level)):
			block1 = convInLRelu(inlayer, n_in_maps=inlayer.shape.as_list()[4], n_out_maps= n_out_maps)
			output_shape.append(block1.shape.as_list())
			n_out_maps = n_out_maps/2
		with tf.variable_scope('dec_level{}b2'.format(level)):
			block_less_maps = convInLRelu(block1, n_in_maps=block1.shape.as_list()[4], n_out_maps= n_out_maps)
			output_shape.append(block_less_maps.shape.as_list())
		decoded = upsample(block_less_maps)
		output_shape.append(decoded.shape.as_list())
		decoded_concat = tf.concat([decoded, level_out[n_encoding-(level+1)]], 4)
		output_shape.append(decoded_concat.shape.as_list())
		inlayer = decoded_concat

	with tf.variable_scope('finalb1'):	
		block1 = convInLRelu(inlayer, n_in_maps=inlayer.shape.as_list()[4], n_out_maps= n_out_maps)
		output_shape.append(block1.shape.as_list())
	with tf.variable_scope('finalb2'):
		block2 = convInLRelu(block1, n_in_maps=block1.shape.as_list()[4], n_out_maps= n_out_maps)
		output_shape.append(block2.shape.as_list())
		inlayer = block2
	y_pred = final_layer(inlayer, n_in_maps=inlayer.shape.as_list()[4], n_out_maps= OUT_LABELs)
	output_shape.append(y_pred.shape.as_list())

	if output_check:
		for i,shape in enumerate(output_shape):
			print("op ",i,"Shape",shape)
		
	return y_pred


