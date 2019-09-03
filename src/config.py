import tensorflow as tf
from collections import OrderedDict
VISIBLE_GPUS = '2'
import os 

#Data Config
BRATS_DATA_PATH = '/tic/Uday/Data/brats18/training/'
TRAIN_PERCENTAGE = 0.7
TEST_PERCENTAGE = 0.15
NO_OF_MODALITIES = 4
NO_OF_LABELS = 4
CROP_SHAPE_TO = (256,256,160)
SELECT_ONLY_VALID_PATCHES = True
LABEL_PIXELS_THRESHOLD = 100
DATA_LABELS_DICT = OrderedDict([('Everything else',0),('Tumor core',1),('Peritumoral Edema ',2),('Enhancing Tumor',4 )])


# Training Config
TRAINING_TAG = 'test'
TRAINING_SAMPLES_TO_USE = 1
NO_OF_EPOCHS = 2
BATCH_SIZE = 2
PATCH_SIZE = (128,128,128)
PATCH_STRIDE = (128,128,32)
TO_SHUFFLE = True
ADAM_LEARNING_RATE = 1e-4
PRINT_MODEL_SUMMARY = False

#Model Config

nnunet_IN_CHANNELS = NO_OF_MODALITIES
nnunet_BASE_FILTER = [3,3,3]
nnunet_FIRST_OUT_CHANNEL = 30
nnunet_CONV_STRIDE = [1,1,1,1,1] 
nnunet_leaky_relu_alpha = 0.2
nnunet_OUT_LABELs = NO_OF_LABELS
nnunet_DTYPE = tf.float32 
nnunet_initialization_std = 0.1
nnunet_NO_OF_ENCODING_LAYERS = 4

# Test Config
SAVED_MODELS_PATH= '/tic/Uday/TumorSeg/sandbox/t_uday_nnUnet-tf/saved_model/best_model/'

SAVED_CKPT = os.path.join(SAVED_MODELS_PATH,"all_128p_nn_unet.ckpt")
TEST_PATCH_SIZE = (256,256,160)
TEST_PATCH_STRIDE = (128,128,32)
TEST_TO_SHUFFLE = False
TEST_BATCH_SIZE = 1

#Validation Config

VAL_PATCH_SIZE = (128,128,128)
VAL_PATCH_STRIDE = (128,128,32)
VAL_TO_SHUFFLE = False
VAL_BATCH_SIZE = 1

def write_config_to_file(f):
	f.write("No. of Training samples: 170\n")
	f.write(str(DATA_LABELS_DICT))
	f.write("\n Epochs %s \n" %(NO_OF_EPOCHS))
	f.write("Path size: " )
	f.write(" ".join(str(PATCH_SIZE)))
	f.write("\n Path stride: " )
	f.write(" ".join(str(PATCH_STRIDE)))
	f.write("\n --------------- \n" )