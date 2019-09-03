import tensorflow as tf 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from data import Brats_Data_generator, generate_partitions
import os
from nnunetmodel import nnunet
from metric_utils import multiclass_dice_loss, label_wise_metrics, dsc_labels
from tqdm import tqdm
from tqdm import trange
import config as cf
import numpy as np
#from validate import validate_brats

os.environ["CUDA_VISIBLE_DEVICES"]= cf.VISIBLE_GPUS 
data_path = cf.BRATS_DATA_PATH
no_epochs = cf.NO_OF_EPOCHS
tag = cf.TRAINING_TAG
#Create partitions of data folders

partition, folder_label = generate_partitions(data_path, train_percentage=cf.TRAIN_PERCENTAGE, test_percentage=cf.TEST_PERCENTAGE)

#-----------------------------------------------

partition['train'] = partition['train'][:cf.TRAINING_SAMPLES_TO_USE]
partition['validation'] = partition['validation'][:cf.TRAINING_SAMPLES_TO_USE]
#------------------------------------------------

print('Number of data points used for training: ',len(partition['train']))
train_data_generator = Brats_Data_generator(partition['train'], in_shape=cf.CROP_SHAPE_TO, batch_size = cf.BATCH_SIZE, patch_size = cf.PATCH_SIZE, patch_stride=cf.PATCH_STRIDE, n_channels = cf.NO_OF_MODALITIES, n_labels = cf.NO_OF_LABELS, shuffle=cf.TO_SHUFFLE, only_valid_patches = cf.SELECT_ONLY_VALID_PATCHES)
validation_data_generator = Brats_Data_generator(partition['validation'], in_shape=cf.CROP_SHAPE_TO, batch_size = cf.VAL_BATCH_SIZE, patch_size = cf.VAL_PATCH_SIZE, patch_stride=cf.VAL_PATCH_STRIDE, n_channels = cf.NO_OF_MODALITIES, n_labels = cf.NO_OF_LABELS, shuffle=cf.VAL_TO_SHUFFLE, only_valid_patches = False)

no_training_samples = train_data_generator.no_batches_per_epoch()
print('Number of Training Samples (batch of patches):', no_training_samples)

#======Build Comp Graph==========

x = tf.placeholder(dtype=tf.float32,shape=[None,*cf.PATCH_SIZE,cf.NO_OF_MODALITIES])
#x = tf.placeholder(dtype=tf.float32,shape=[None,None, None, None,cf.NO_OF_MODALITIES])
y_ = tf.placeholder(dtype=tf.float32,shape=[None,*cf.PATCH_SIZE,cf.NO_OF_LABELS])
x_v = tf.placeholder(dtype=tf.float32,shape=[None,*cf.VAL_PATCH_SIZE,cf.NO_OF_MODALITIES])
y_v = tf.placeholder(dtype=tf.float32,shape=[None,*cf.VAL_PATCH_SIZE,cf.NO_OF_LABELS])
y_pred = nnunet(x, output_check=cf.PRINT_MODEL_SUMMARY) 																#nparray

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ ,logits=y_pred))
loss= multiclass_dice_loss(y_, y_pred)
#loss = cross_entropy
train_step = tf.train.AdamOptimizer(cf.ADAM_LEARNING_RATE).minimize(loss)
#========== Metric ==============

#correct_prediction = tf.equal(tf.argmax(y_pred, 4), tf.argmax(y_,4))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

dsc = dsc_labels(y_, y_pred) 

saver = tf.train.Saver()
loss_epoch_list = []
val_loss_epoch_list = []
acc_epoch_list = []
val_acc_epoch_list = []


def validate_brats(sess, validation_data_generator):
    """
    Validates the model after every epoch
    ----------------
    Params: 
    sess: Give the tensorflow session that is training the model
    validation_data_generator: The data generator for the validation data
    ----------------
    Val Mean edema dsc, Val loss
    """
    ed_dsc_list = []
    loss_list = []
    no_validation_samples = validation_data_generator.no_batches_per_epoch()

    for i in range(no_validation_samples):
        bg_dsc = 0
        core_dsc = 0
        ed_dsc = 0
        enh_dsc = 0 

        #---------- Generate a batch ------------
        X_val, y_val= validation_data_generator.get_a_batch(i)
        print("y val shape:", y_val.shape)
        print("x val shape:", X_val.shape)
        #---------- Predict ----------------------
        y_predic = sess.run(y_pred, feed_dict={x: X_val, y_: y_val})
        #---------- Calculate metrics ------------
        cf.BATCH_SIZE = 1
        loss_val = sess.run(loss, feed_dict={x: X_val, y_: y_val})
        #loss_val = 0
        bg_dsc, core_dsc, ed_dsc, enh_dsc = label_wise_metrics(y_val, y_predic)
        ed_dsc_list.append(ed_dsc)
        loss_list.append(loss_val)
        cf.BATCH_SIZE = 2

    return sum(ed_dsc_list)/len(ed_dsc_list), sum(loss_list)/len(loss_list)


#======= Training ===============
with open("./results/"+tag+"_Training_log.txt","w") as f:
    cf.write_config_to_file(f)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(no_epochs):
            print("Epoch ", i)
            print("-----------")
            train_data_generator.shuffle_at_end_of_epoch()
            loss_batch_list = []
            ed_dsc_batch_list = []
            with trange(no_training_samples) as t:
                for sample in t:
                    #------------Generate train batches ----------
                    X_train, y_train = train_data_generator.get_a_batch(sample)
                    #------------ Weight updates--------------------------
                    sess.run(train_step, feed_dict={x: X_train, y_: y_train})  
                    #------------ Prediction and metric calculation
                    y_predicted = sess.run(y_pred, feed_dict={x: X_train, y_: y_train})
                    loss_val = sess.run(loss, feed_dict={x: X_train, y_: y_train})
                    bg_dsc, core_dsc, ed_dsc, enh_dsc = label_wise_metrics(y_train, y_predicted)
                    loss_batch_list.append(loss_val)
                    ed_dsc_batch_list.append(ed_dsc)
                    t.set_postfix(loss=loss_val, core_dsc=core_dsc, ed_dsc=ed_dsc)
                    # Write to text file
                    if sample%10==0:
                        f.write("Epoch %s iter: %s training loss: %s edema dsc: %s core dsc: %s \n" %(i,sample,loss_val, ed_dsc, core_dsc))

                #--------Validation---------
                val_ed_dsc, val_loss = validate_brats(sess, validation_data_generator)

                print("Max train loss: ", max(loss_batch_list), "|| Min train loss: ", min(loss_batch_list),"|| Mean train Loss: ", sum(loss_batch_list)/len(loss_batch_list))
                print("Max train ed dsc: ", max(ed_dsc_batch_list), "|| Min train ed dsc: ", min(ed_dsc_batch_list),"|| Mean train ed dsc: ", sum(ed_dsc_batch_list)/len(ed_dsc_batch_list))
                print("Mean validation loss: ", val_loss)
                print("Mean validation Edema dsc: ", val_ed_dsc)
                f.write("Max train loss: %s || Min train loss: %s || Mean train Loss: %s \n" %(max(loss_batch_list),min(loss_batch_list), sum(loss_batch_list)/len(loss_batch_list)))
                f.write("Max train ed_dsc: %s || Min train ed_dsc: %s || Mean train ed_dsc: %s \n" %(max(ed_dsc_batch_list),min(ed_dsc_batch_list), sum(ed_dsc_batch_list)/len(ed_dsc_batch_list)))
                f.write("Mean validation loss: %s \n Mean validation Edema dsc: %s " %(val_loss, val_ed_dsc))
                f.write("----------------------\n")

            loss_epoch_list.append(sum(loss_batch_list)/len(loss_batch_list))
            acc_epoch_list.append(sum(ed_dsc_batch_list)/len(ed_dsc_batch_list))
            val_loss_epoch_list.append(val_loss)
            val_acc_epoch_list.append(val_ed_dsc)
        saver.save(sess, "./saved_model/"+tag+"_nn_unet.ckpt")
# Print plots and save them
plt.figure(1)

plt.plot(acc_epoch_list,label='train_dsc')
plt.plot(val_acc_epoch_list,label='val_dsc')
plt.title('Edema dice score')
plt.ylabel('Dsc')
plt.xlabel('iters')
plt.legend(loc='upper left')
plt.savefig('./results/'+tag+'_acc_plot.png')


plt.figure(2)
plt.plot(loss_epoch_list,label='train_loss')
plt.plot(val_loss_epoch_list,label='val_loss')
plt.title('Model loss')
plt.ylabel('loss')
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.savefig('./results/'+tag+'_loss_plot.png')

    	


