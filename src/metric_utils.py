import numpy as np 
import tensorflow as tf 
import config as cf
def multiclass_dice_loss(true_label, pred_label, epsilon = 1e-8):
	"""
	Calculates the multi class dice loss 
	------------
	Params: 
	true_label:One hot encoded true seg.  Np array, shape (batch,x,y,z,labels)
	pred_label:Softmax output of network. Np array, shape (batch,x,y,z,labels)
	true_label and pred_label must have the same shapes
	------------
	Output:
	Multiclass dice loss
	"""
	#batches = pred_label.shape.as_list()[0]
	#channels = pred_label.shape.as_list()[4]
	channels = 4
	true_label = tf.cast(true_label, tf.float32)
	pred_label = tf.cast(pred_label, tf.float32)
	_loss = 0
	for_test = 0
	for batch in range(cf.BATCH_SIZE):
		for label in range(channels):
			numerator = tf.reduce_sum(tf.multiply(true_label[batch,:,:,:,label],pred_label[batch,:,:,:,label]))
			denominator = tf.reduce_sum(true_label[batch,:,:,:,label]) + tf.reduce_sum(pred_label[batch,:,:,:,label])
			_loss = _loss + (numerator + epsilon)/(denominator + epsilon)
	_loss = 1 - (2*_loss/channels)
	#return tf.reduce_mean(_loss)
	return _loss

def dsc_labels(true_label, pred_label, epsilon = 1e-8):
	"""
	Gets dice loss only for tumor labels and not background
	To understand how well the tumor is being learnt
	"""

	los= multiclass_dice_loss(true_label[:,:,:,1:3],pred_label[:,:,:,1:3])
	dsc = (1 - los)*(3/2)

	return dsc

def label_wise_metrics(true_label, pred_label):
	"""
	Calculates metrics label wise
	"""
	batches = pred_label.shape.as_list()[0]
	channels = pred_label.shape.as_list()[4]
	bg_dsc = [None]*batches
	bg_f1 = [None]*batches
	core_dsc = [None]*batches
	core_f1 = [None]*batches
	ed_dsc = [None]*batches
	ed_f1 = [None]*batches
	enh_dsc = [None]*batches
	enh_f1 = [None]*batches

	for batch in range(batches):
		bg_dsc[batch], bg_f1[batch] = spatial_overlap_metrics(true_label[batch,:,:,:,0], pred_label[batch,:,:,:,0])
		core_dsc[batch], core_f1[batch] =  spatial_overlap_metrics(true_label[batch,:,:,:,1], pred_label[batch,:,:,:,1])
		ed_dsc[batch], ed_f1[batch] =  spatial_overlap_metrics(true_label[batch,:,:,:,2], pred_label[batch,:,:,:,2])
		enh_dsc[batch], enh_f1[batch] =  spatial_overlap_metrics(true_label[batch,:,:,:,3], pred_label[batch,:,:,:,3])

	bgdsc = sum(bg_dsc)/len(bg_dsc)
	bgf1 = sum(bg_f1)/len(bg_f1)
	coredsc = sum(core_dsc)/len(core_dsc)
	coref1 = sum(core_f1)/len(core_f1)
	eddsc = sum(ed_dsc)/len(ed_dsc)
	edf1 = sum(ed_f1)/len(ed_f1)
	enhdsc = sum(enh_dsc)/len(enh_dsc)
	enhf1 = sum(enh_f1)/len(enh_f1)

	return bgdsc, coredsc, eddsc, enh_dsc

def f_spatial_overlap_cardinalities(true_label, pred_label):
	"""
	Calculates the TP, FP, TN, FN for fuzzy segmentations 
	Foreground and background are the only two classes being considered
	This should be used for class wise metrics
	-------------
	Params: 
	true_label:One hot encoded true seg.  Np array, shape (batch,x,y,z,labels)
	pred_label:Softmax output of network. Np array, shape (batch,x,y,z,labels)
	true_label and pred_label must have the same shapes
	--------------
	Output:
	true_positives, false_positives, true_negatives, false_negatives
	"""
	temp_tp = np.minimum(true_label,pred_label)
	true_positives = np.sum(temp_tp)

	temp_fp = np.maximum(pred_label - true_label,0)
	false_positives = np.sum(temp_fp)

	true_label_bg = np.subtract(1, true_label)
	pred_label_bg = np.subtract(1, pred_label)

	temp_tn = np.minimum(true_label_bg, pred_label_bg)
	true_negatives = np.sum(temp_tn)

	temp_fn = np.maximum(pred_label_bg - true_label_bg,0)
	false_negatives = np.sum(temp_fn)

	return true_positives, false_positives, true_negatives, false_negatives

def spatial_overlap_metrics(true_label, pred_label):
	"""
	Calculates the Dice score and F1 score for two class scenario
	------------
	Params: 
	true_label:One hot encoded true seg.  Np array, shape (batch,x,y,z,labels)
	pred_label:Softmax output of network. Np array, shape (batch,x,y,z,labels)
	true_label and pred_label must have the same shapes
	------------
	Output:
	dicescore, f1_measure
	"""

	tp,fp,tn,fn = f_spatial_overlap_cardinalities(true_label, pred_label)

	dsc = (2*tp)/((2*tp) + fp + fn)

	#f1_measure = f1_score(true_label.flatten(), pred_label.flatten())
	f1_measure = 0
	return dsc, f1_measure
