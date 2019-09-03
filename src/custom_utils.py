import numpy as np 
import random
import config as cf
def resize_by_padding(volume, size_to_reshape=(256,256,256)):
	"""
	Resizes the given volume by zero padding to get the size_to_reshape
	---------------------
	Params: 
	volume - Numpy array
	size_to_reshape - Tuple of size
	---------------------
	Output:
	Numpy array with the shape of size_to_reshape having the appropriate zero padding
	"""

	in_shape = list(volume.shape)
	out_shape = list(size_to_reshape)


	if out_shape[0]<in_shape[0] or out_shape[1]<in_shape[1] or out_shape[2]<in_shape[2]:
		raise Exception("The output shape should be greater than the input shape")
	to_pad = []
	for i in range(len(in_shape)):
		to_pad.append((out_shape[i]-in_shape[i])//2)


	reshaped_volume = np.zeros(out_shape, dtype = np.float32)
	reshaped_volume[to_pad[0]:to_pad[0]+in_shape[0],to_pad[1]:to_pad[1]+in_shape[1],to_pad[2]:to_pad[2]+in_shape[2]] = volume

	return reshaped_volume

def extract_volume(volume, shape_to_extract = (128,128,128)):
	"""
	Extracts a fraction of the input volume 
	-----------------
	Params:
	volume - Numpy array
	shape_to_extract - tuple of req shape
	-----------------
	Output:
	Numpy array of extracted volume
	"""
	in_shape = list(volume.shape)
	out_shape = list(shape_to_extract)
	if out_shape[0]>in_shape[0] or out_shape[1]>in_shape[1] or out_shape[2]>in_shape[2]:
		raise Exception("The requested volume to be extracted is bigger than the input volume")	
	extracted_volume = volume[:out_shape[0], :out_shape[1], :out_shape[2]]

	return extracted_volume
def _no_of_patches(in_shape, patch_size, stride_size):
	"""
	Get the number of patches that can be extracted from the volume given the patch size and stride size
	-------------------
	Params:
	in_shape - Shape of the input volume as a List
	patch_size - Size of patch as a List
	stride_size - Stride as a list
	-------------------
	Output: no of patches as a list
	"""
	no_of_patches = []

	for i in range(len(in_shape)):
		no_of_patches.append(((in_shape[i] - patch_size[i])//stride_size[i]) + 1)

	return no_of_patches
def generate_patch_volumes(volume, patch_size=(128,128,128), stride_size=(32,32,32)):
	"""
	Given a 3d volume, creates a list of patches according to the patch size and the stride size given as an input
	-------------------
	Params:
	volume: Numpy array
	patch_size: tuple indicating the size of patches to extract
	stride_size: tuple indicating the dim to moven when extracting the pathces
	-------------------
	Output:

	"""

	in_shape = list(volume.shape)
	patch_size = list(patch_size)
	stride_size = list(stride_size)

	no_of_patches = _no_of_patches(in_shape, patch_size, stride_size)

	list_of_patches = []
	
	temp_vol = volume
	for i in range(no_of_patches[0]):
		for j in range(no_of_patches[1]):
			for k in range(no_of_patches[2]):

				temp_extracted = extract_volume(temp_vol,shape_to_extract = patch_size)
				list_of_patches.append(temp_extracted)
				temp_vol = volume[stride_size[0]*i:, stride_size[1]*j:, stride_size[2]*k:]
				
	return list_of_patches


def generate_patch_startid_list(volume_shape=(256,256,160), patch_size=(128,128,128), stride_size=(32,32,32)):
	"""
	Given a 3d volume, creates a list of patches according to the patch size and the stride size given as an input
	-------------------
	Params:
	volume: tuple indicating the shape of volume
	patch_size: tuple indicating the size of patches to extract
	stride_size: tuple indicating the dim to moven when extracting the pathces
	-------------------
	Output:
	List containing the start ids (tuples) of all possible patches
	"""

	in_shape = list(volume_shape)
	patch_size = list(patch_size)
	stride_size = list(stride_size)

	no_of_patches = _no_of_patches(in_shape, patch_size, stride_size)

	#list_of_patches = []
	list_patch_startidx = []
	#temp_vol = volume
	for i in range(no_of_patches[0]):
		for j in range(no_of_patches[1]):
			for k in range(no_of_patches[2]):

				#temp_extracted = extract_volume(temp_vol,shape_to_extract = patch_size)
				#list_of_patches.append(temp_extracted)
				#temp_vol = volume[stride_size[0]*i:, stride_size[1]*j:, stride_size[2]*k:]
				temp_start_idx = (stride_size[0]*i, stride_size[1]*j, stride_size[2]*k)
				list_patch_startidx.append(temp_start_idx)
	#return list_of_patches, list_patch_startidx
	return list_patch_startidx



def get_patch(volume, patch_size, patch_start_idx):
	"""
	Return the patch of volume 
	---------------
	Params:
	volume: Np array of input vol
	patch_size: tuple of patch size
	patch_start_idx: The starting voxel of the patch
	--------------
	Output:
	Np array of volume of the requested patch
	"""

	temp_vol = volume[patch_start_idx[0]:, patch_start_idx[1]:, patch_start_idx[2]:]
	patch_volume = extract_volume(temp_vol, shape_to_extract = patch_size)
	return patch_volume

def convert_to_one_hot(volume, labels=cf.DATA_LABELS_DICT):
	"""
	Converts the label map to a on hot encoded matrix
	---------------
	Params:
	Volume: Np array of label map
	Labels: Dict of class names and values in the label map
	---------------
	Output:
	One hot encoded matrix with shape (*dim, channels)
	"""
	#print("Label convention: ",labels)
	no_channels = len(labels)
	one_hot_label = np.zeros((*volume.shape, no_channels))

	for i,label in enumerate(labels):
		#if labels[label]!=0:
		one_hot_label[volume.astype(int)==labels[label],i] = 1

	return one_hot_label

def is_valid_patch(volume,no_label_pixels = cf.LABEL_PIXELS_THRESHOLD):
	"""
	To know if a patch of volume is valid. The condition to satisfy for a valid patch is 
	non zero elements of volume should be >= LABEL_PIXELS_THRESHOLD
	-------------------
	Output:
	True or False
	"""

	non_zero_voxels = np.count_nonzero(volume)
	if non_zero_voxels<no_label_pixels:
		return False

	return True








