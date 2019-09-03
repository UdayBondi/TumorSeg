import nibabel as nib
import numpy as np
import os
from custom_utils import resize_by_padding, generate_patch_startid_list, get_patch, convert_to_one_hot, is_valid_patch
from tqdm import tqdm
from tqdm import trange
import config as cf
def load_nii(file_path,to_array=True):
	"""
	Fucntion load_nii:
	------------------------------
	Loads a .nii file into an array
	------------------------------
	Params: 
	file_path: Give the path to the file. 
		Note: Use os.join.path to reference to the file.
	to_array:
		True: <np array>
		False: <nibabel nifti image>
	------------------------------
	Output: Nibabel img object
	"""
	img = nib.load(file_path)

	if to_array:
		img_array = nib.load(file_path).get_fdata()
		return img_array
	
	return img

def normalize_data(img_array):
	"""
	Function normalize_data:
	-------------------------
	** Assumption: Skull Stripped Data **
	Subtract by mean and divide by std of the brain region only

	This would make sure that the brain region(non zero) values have a mean of 0 and std as 1
	-------------------------
	Params: image as a nunmpy array
	-------------------------
	Output: normalized numpy array
	"""
	brain = np.nonzero(img_array)
	img_array_non_zero = img_array[brain]
	mean_brain = np.mean(img_array_non_zero)
	std_brain = np.std(img_array_non_zero)
	img_array[brain] = (img_array[brain] - mean_brain)/std_brain

	return img_array

def create_data_list(train_data_path):
	"""
	Creates lists of HGG and LGG files names from Brats data
	--------------------
	Param: Path to the training data. The path should lead to two folder HGG and LGG
	--------------------
	Output: Two lists. HGG list and LGG list.
	"""
	global path
	HGG_dir_list = next(os.walk(train_data_path + 'HGG/'))[1]
	LGG_dir_list = next(os.walk(train_data_path + 'LGG/'))[1]
	path = train_data_path
	return HGG_dir_list, LGG_dir_list

def generate_partitions(train_data_path, train_percentage = 0.7, test_percentage = 0.15):
	"""
	Generates a dictionary with keys: train, validation, test. File names are the values of the dictionary. 
	The file names are shuffled before splitting into train, test and validation.

	----------------------
	Params: train_data_path: Path to the training data. The path should lead to two folders HGG and LGG.
	train_percentage (default = 0.7) and test_percentage (default = 0.15)
	----------------------
	Outputs:
	Partition: Dictionary 
	Labels: List that takes in a file name and return a label (HGG = 0 and LGG = 1)

	"""
	global folderName                   # list that gives takes in a file name and gives the type (HGG/LGG) it belongs to 
	HGG_dir_list, LGG_dir_list = create_data_list(train_data_path)
	completelist = HGG_dir_list + LGG_dir_list
	np.random.shuffle(completelist)                         # shuffles in place

	partition = {}
	labels = {}
	folder_name = {}
	# Train , Validation, test 
	partition['test'] = completelist[0:int(len(completelist) * test_percentage)]
	trainlist = completelist[int(len(completelist) * test_percentage):len(completelist)]

	partition['train'] = trainlist[0:int(len(trainlist) * train_percentage)]
	partition['validation'] = trainlist[int(len(trainlist) * train_percentage):len(trainlist)]


	# HGG=0
	# LGG=1
	for directory in HGG_dir_list:
		labels[directory] = 0
		folder_name[directory] = 'HGG'
	for directory in LGG_dir_list:
		labels[directory] = 1
		folder_name[directory] = 'LGG'
	folderName = folder_name

	return partition, labels
	
class Brats_Data_generator():

	def __init__(self, list_IDs, in_shape=(256,256,160), batch_size = 2, patch_size = (128,128,128), patch_stride=(32,32,32), n_channels = 4, n_labels = 4, shuffle=True, only_valid_patches= True,mode='train'):

		self.list_IDs = list_IDs
		self.mode = mode
		self.in_shape = in_shape
		self.batch_size = batch_size
		self.patch_size = patch_size
		self.patch_stride = patch_stride
		self.n_channels = n_channels
		self.n_labels = n_labels
		self.shuffle = shuffle
		self.only_valid_patches = only_valid_patches
		self.index_list = self.create_patch_index_list(self.only_valid_patches)
		self.no_samples = len(self.index_list)
		self.shuffle_at_end_of_epoch()

	def no_batches_per_epoch(self):
		"""
		Denotes the number of batches per epoch
		"""
		return int(np.floor(self.no_samples / self.batch_size))

	def create_patch_index_list(self, only_valid_patches):
		"""
		Creates a list containing tuples of (patientid, patchstart_x, patchstart_y, patchstart_z)
		This list can be used to 
		"""
		list_of_patch_startids = generate_patch_startid_list(self.in_shape, self.patch_size, self.patch_stride)
		
		index_list = []
		if only_valid_patches:
			print("Generating_valid_patches.....\n")
		for patient_id in tqdm(self.list_IDs):
			for patch_start in list_of_patch_startids:
				if only_valid_patches:
					seg_patch = self.get_seg_label_patch(patient_id, patch_start)
					if is_valid_patch(seg_patch):
						index_list.append((patient_id, *patch_start))
				else:
					index_list.append((patient_id, *patch_start))
		return index_list

	def shuffle_at_end_of_epoch(self):
		"""
		Creates a list of indexes for iterating and accesing list of IDs
		The list of IDs is shuffled after every epoch (by shuffling the index list). This would ensure stochasticity. 
		This also means you can simply iterate using an index will getting batches
		"""

		#self.indexes = np.arrange(len(self.list_IDs))

		if self.shuffle==True:
			np.random.shuffle(self.index_list)

	def get_a_batch(self, index):

	    # Generate indexes of the batch
	    indexes_temp = self.index_list[index * self.batch_size:(index + 1) * self.batch_size]

	    if self.mode == 'test':
	    	X, y,original_images = self.data_generation(indexes_temp)
	    	# return X, [y1,y2]
	    	return X, y, original_images
	    # return X, [y1,y2]
	    X, y = self.data_generation(indexes_temp)
	    return X, y

	def data_generation(self, indexes_temp):
		"""
		"""

		# Initialization
		X = np.empty((self.batch_size, *self.patch_size, self.n_channels))
		y = np.empty((self.batch_size, *self.patch_size, self.n_labels))

		#Generate data
		for i, id_with_patchid in enumerate(indexes_temp):

			ID = id_with_patchid[0]
			#print(ID)
			patch_start_idx = id_with_patchid[1:]
			# Get paths
			img1_path = path + folderName[ID] + '/' + ID + '/' + ID + '_flair.nii.gz'
			img2_path = path + folderName[ID] + '/' + ID + '/' + ID + '_t1.nii.gz'
			img3_path = path + folderName[ID] + '/' + ID + '/' + ID + '_t1ce.nii.gz'
			img4_path = path + folderName[ID] + '/' + ID + '/' + ID + '_t2.nii.gz'
			img5_path = path + folderName[ID] + '/' + ID + '/' + ID + '_seg.nii.gz'

			# Load data
			iflair = load_nii(img1_path)
			it1 = load_nii(img2_path)
			it1ce = load_nii(img3_path)
			it2 = load_nii(img4_path)
			iseg = load_nii(img5_path)

			# Change Shape with zero padding

			iflair = resize_by_padding(iflair, self.in_shape)
			it1 = resize_by_padding(it1, self.in_shape)
			it1ce = resize_by_padding(it1ce, self.in_shape)
			it2 = resize_by_padding(it2, self.in_shape)
			iseg = resize_by_padding(iseg, self.in_shape)

			if self.mode =='test':
				original_images = np.stack((iflair,it1,it1ce,it2,iseg),axis = 3)

			# Normalize the data 

			iflair = normalize_data(iflair)
			it1 = normalize_data(it1)
			it1ce = normalize_data(it1ce)
			it2 = normalize_data(it2)

			#Get patches

			iflair_patch = get_patch(iflair, self.patch_size, patch_start_idx)
			it1_patch = get_patch(it1, self.patch_size, patch_start_idx)
			it1ce_patch = get_patch(it1ce, self.patch_size, patch_start_idx)
			it2_patch = get_patch(it2, self.patch_size, patch_start_idx)
			iseg_patch = get_patch(iseg, self.patch_size, patch_start_idx)
			images = np.stack((iflair_patch,it1_patch,it1ce_patch,it2_patch),axis = 3)
			true_label = iseg_patch

			# Convert to one hot enoding

			one_hot_label = convert_to_one_hot(true_label)

			X[i, ] = images
			y[i, ] = one_hot_label

		if self.mode =='test':
			return X, y, original_images

		return X,y
	def get_seg_label_patch(self,patientid, patch_start_idx):

		ID = patientid
		seg_path = path + folderName[ID] + '/' + ID + '/' + ID + '_seg.nii.gz'

		iseg = load_nii(seg_path)
		iseg = resize_by_padding(iseg, self.in_shape)
		iseg_patch = get_patch(iseg, self.patch_size, patch_start_idx)

		return iseg_patch







		

















































	
