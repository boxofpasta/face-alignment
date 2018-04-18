# https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py
from os import path, sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import datetime
import time
import scipy.misc
import numpy as np
import copy
import matplotlib
import keras
import src.utils.helenUtils as helenUtils
import src.utils.generalUtils as utils
from skimage.transform import resize
import cv2
import json
import time
from src.model import getPointMaskerRefined
from src.BatchGenerator import PointMaskCascadedBatchGenerator
import keras.backend as K
from keras.utils import plot_model
from keras.models import load_model
from keras.applications import mobilenet
from tensorflow.python import debug as tf_debug

# config
np.random.seed(0)
ibug_version = False 
train_on_all = False  # true if no validation
notify_training_complete = True
project_root_path = '../'
train_path =  project_root_path + 'data/train/'
val_path = project_root_path + 'data/train/'
samples = ['100466187_1', '13602254_1', '2908549_1', '100032540_1', '1691766_1', '11564757_2', '110886318_1']
im_side_len = 224
mask_side_len = 28
train_all_epochs = 240
train_partial_epochs = 180
model_name = 'point_masker_refined'

with open(train_path + 'names.json') as fp:
    all_names = json.load(fp)

model = getPointMaskerRefined(im_side_len, mask_side_len)
now = datetime.datetime.now()
time_str = now.strftime("%m-%d:%H:%M")
model_folder = project_root_path + 'saved-models/' + model_name + '/' + time_str + '/'
model_path = model_folder + 'model.h5'
val_split_ratio = 0.0 if train_on_all else 0.3
num_val_samples = int(val_split_ratio * len(all_names))
val_indices = np.random.choice(len(all_names), num_val_samples, replace=False)
all_val_names = [all_names[ind] for ind in val_indices]
all_train_names = list(set(all_names) - set(all_val_names))
print 'Number of train images: ' + str(len(all_train_names))
print 'Number of val images: ' + str(len(all_val_names))
print 'Number of images in total: ' + str(len(all_names))

train_batch_generator = PointMaskCascadedBatchGenerator(all_train_names, train_path, mask_side_len,
                                                                im_side_len, augment_on_generate=True, ibug_version=ibug_version)
val_batch_generator = PointMaskCascadedBatchGenerator(all_val_names, val_path, mask_side_len, 
                                                                im_side_len, augment_on_generate=False, ibug_version=ibug_version)

epochs = train_all_epochs if train_on_all else train_partial_epochs
monitor_str = 'loss' if train_on_all else 'val_loss'

# tensorboard
tb_log_dir = model_folder + 'tensorboard/'
tb_callback = keras.callbacks.TensorBoard(log_dir=tb_log_dir, 
                                            histogram_freq=0,
                                            write_grads=True,
                                            write_graph=True, 
                                            write_images=True)

# dynamic learning rate and checkpoints
lr_callback = keras.callbacks.ReduceLROnPlateau(monitor=monitor_str, factor=0.5, patience=10, verbose=1, mode='auto', 
                                            epsilon=0.0001, cooldown=0, min_lr=0)
cp_callback = keras.callbacks.ModelCheckpoint(model_path, monitor=monitor_str, verbose=1, save_best_only=True, 
                                            save_weights_only=False, mode='auto', period=5)

# train
val_data = None if train_on_all else val_batch_generator.getAllData()
model.fit_generator(generator=train_batch_generator.generate(),
                    validation_data=val_data,  
                    steps_per_epoch=train_batch_generator.steps_per_epoch,
                    epochs=epochs,
                    callbacks=[tb_callback, cp_callback, lr_callback])
print 'Finished training for: ' + str(epochs) + ' epochs, everything was saved to: ' + model_path