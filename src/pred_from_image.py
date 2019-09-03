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
import nibabel as nib

os.environ["CUDA_VISIBLE_DEVICES"]= cf.VISIBLE_GPUS 
data_path = cf.BRATS_DATA_PATH
no_epochs = cf.NO_OF_EPOCHS

#Create partitions of data folders

partition, folder_label = generate_partitions(data_path, train_percentage=cf.TRAIN_PERCENTAGE, test_percentage=cf.TEST_PERCENTAGE)

#-----------------------------------------------
#Taking only 20 samples for now 
partition['test'] = partition['test'][:20]
#------------------------------------------------

print('Number of data points used for testing: ',len(partition['test']))
test_data_generator = Brats_Data_generator(partition['test'], in_shape=cf.CROP_SHAPE_TO, batch_size = cf.TEST_BATCH_SIZE, patch_size = cf.TEST_PATCH_SIZE, patch_stride=cf.TEST_PATCH_STRIDE, n_channels = cf.NO_OF_MODALITIES, n_labels = cf.NO_OF_LABELS, shuffle=cf.TEST_TO_SHUFFLE, only_valid_patches = False, mode='test')
#validation_data_generator = Brats_Data_generator(partition['validation'], in_shape=cf.CROP_SHAPE_TO, batch_size = len(partition['validation']), patch_size = cf.CROP_SHAPE_TO, patch_stride=cf.PATCH_STRIDE, n_channels = cf.NO_OF_MODALITIES, n_labels = cf.NO_OF_LABELS, shuffle=cf.TO_SHUFFLE)
no_test_samples = test_data_generator.no_batches_per_epoch()
print('Number of Training Samples (batch of patches):', no_test_samples)

#======Build Comp Graph==========

x = tf.placeholder(dtype=tf.float32,shape=[None,*cf.TEST_PATCH_SIZE,cf.NO_OF_MODALITIES])
y_ = tf.placeholder(dtype=tf.float32,shape=[None,*cf.TEST_PATCH_SIZE,cf.NO_OF_LABELS])
y_pred = nnunet(x, output_check=cf.PRINT_MODEL_SUMMARY) 																#nparray

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ ,logits=y_pred))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ ,logits=y_pred))
#loss = cross_entropy
loss= multiclass_dice_loss(y_, y_pred)
train_step = tf.train.AdamOptimizer(cf.ADAM_LEARNING_RATE).minimize(loss)
#========== Metric ==============

#correct_prediction = tf.equal(tf.argmax(y_pred, 4), tf.argmax(y_,4))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
dsc = dsc_labels(y_, y_pred) 

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, cf.SAVED_CKPT)
	for i in range(3):
		X,y, original_images = test_data_generator.get_a_batch(i)
		print("Original_images: ",original_images.shape)
		it2 = nib.Nifti1Image(original_images[:,:,:,3],affine=None)
		iseg = nib.Nifti1Image(original_images[:,:,:,4],affine=None)
		print("X shape: ",X.shape)
		loss_val = 0
		accuracy_val = 0
		#loss_val = sess.run(loss, feed_dict={x: X, y_: y})
		#accuracy_val = sess.run(dsc, feed_dict={x: X, y_: y})
		y_predic = sess.run(y_pred, feed_dict={x: X, y_: y})
		print(y_predic.shape)
		y_predic = np.reshape(np.argmax(y_predic, 4),(256,256,160))
		y_test = np.zeros((256,256,160,3)).astype(np.uint8)
		y_test[y_predic.astype(int)==1] = (128,0,0)		#Maroon
		y_test[y_predic.astype(int)==2] = (255,215,0)	#Gold
		y_test[y_predic.astype(int)==3] = (138,43,226) #Violet

		shape_3d = y_test.shape[0:3]
		rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
		y_test = y_test.copy().view(dtype=rgb_dtype).reshape(shape_3d)

		pred_image = nib.Nifti1Image(y_test,affine=None)
		print("Loss: ",loss_val)
		print("Acc: ",accuracy_val)
		print("y_pred: ", y_predic.shape)
		print("y_pred min: ", y_predic.min())
		print("y_pred max: ", y_predic.max())
		nib.save(pred_image, "4_image_"+str(i)+".nii.gz")
		nib.save(it2, "4_it2_"+str(i)+".nii.gz")
		nib.save(iseg, "4_iseg_"+str(i)+".nii.gz")
		print("-------")