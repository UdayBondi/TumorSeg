import tensorflow as tf 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from data import Brats_Data_generator, generate_partitions
import os
from nnunetmodel import nnunet
from metric_utils import multiclass_dice_loss, dsc_labels
from tqdm import tqdm
from tqdm import trange
import config as cf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]= cf.VISIBLE_GPUS 
data_path = cf.BRATS_DATA_PATH
no_epochs = cf.NO_OF_EPOCHS

#Create partitions of data folders

partition, folder_label = generate_partitions(data_path, train_percentage=cf.TRAIN_PERCENTAGE, test_percentage=cf.TEST_PERCENTAGE)

#-----------------------------------------------
#Taking only 20 samples for now 
partition['train'] = partition['train'][:2]
#------------------------------------------------

print('Number of data points used for training: ',len(partition['train']))
train_data_generator = Brats_Data_generator(partition['train'], in_shape=cf.CROP_SHAPE_TO, batch_size = cf.BATCH_SIZE, patch_size = cf.PATCH_SIZE, patch_stride=cf.PATCH_STRIDE, n_channels = cf.NO_OF_MODALITIES, n_labels = cf.NO_OF_LABELS, shuffle=cf.TO_SHUFFLE, only_valid_patches = cf.SELECT_ONLY_VALID_PATCHES)

#validation_data_generator = Brats_Data_generator(partition['validation'], in_shape=cf.CROP_SHAPE_TO, batch_size = len(partition['validation']), patch_size = cf.CROP_SHAPE_TO, patch_stride=cf.PATCH_STRIDE, n_channels = cf.NO_OF_MODALITIES, n_labels = cf.NO_OF_LABELS, shuffle=cf.TO_SHUFFLE)
no_training_samples = train_data_generator.no_batches_per_epoch()
print('Number of Training Samples (batch of patches):', no_training_samples)

#======Build Comp Graph==========

x = tf.placeholder(dtype=tf.float32,shape=[None,*cf.PATCH_SIZE,cf.NO_OF_MODALITIES])
y_ = tf.placeholder(dtype=tf.float32,shape=[None,*cf.PATCH_SIZE,cf.NO_OF_LABELS])
y_pred = nnunet(x, output_check=cf.PRINT_MODEL_SUMMARY) 																#nparray

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ ,logits=y_pred))
loss, t_value = multiclass_dice_loss(y_, y_pred)
#loss = cross_entropy
train_step = tf.train.AdamOptimizer(cf.ADAM_LEARNING_RATE).minimize(loss)
#========== Metric ==============

correct_prediction = tf.equal(tf.argmax(y_pred, 4), tf.argmax(y_,4))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

dsc = dsc_labels(y_, y_pred) 

saver = tf.train.Saver()
loss_epoch_list = []
acc_epoch_list = []

train_data_generator.shuffle_at_end_of_epoch()

for i in range(7):
	X, y = train_data_generator.get_a_batch(i)

	print("Sum: ", np.sum(y[0,:,:,:,1:3]))
	print("Max x: ", X.max())
	print("Max y: ", y.max())
	print("x: ", X.shape)
	print("y : ", X.shape)



