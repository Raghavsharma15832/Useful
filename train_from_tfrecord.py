import pandas as pd
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow_addons as tfa
tf.compat.v1.disable_eager_execution()

from keras_CNN import Model


def _extract_fn(tfrecord):
    # Extract features using the keys set during creation
    features = {
        'date': tf.io.FixedLenFeature([], tf.string),
        'Image shapes': tf.io.FixedLenFeature([2], tf.int64, default_value= [0,0]),
        'SST': tf.io.FixedLenFeature([], tf.string),
        'SSH': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
       
    }

    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, features)

    sst = tf.io.decode_raw(sample['SST'], tf.float64)
    ssh = tf.io.decode_raw(sample['SSH'], tf.float64)      
    img_shape =sample["Image shapes"]
    label = tf.io.decode_raw(sample['label'], tf.uint8)
    date = sample['date']
    
    

    return [sst,ssh, label, img_shape, date]        

def image_augmentation2(image, label, angles= 0):
    image = tfa.image.rotate(images= image,
                angles= angles ,interpolation= 'BILINEAR',name= None)
    label = tfa.image.rotate(images= label,
                angles=  angles,interpolation= 'BILINEAR',name= None)
    return image, label

def extract_image(iterator, batch_size, angles= 0, training= False):

    one_data = iterator.get_next()
    '''
    one_data is a list of tensor contains [image,  label, img_shape, date]
    image is concatenated form for sst and ssh respectively.
    '''
    image_shape = one_data[3]
    sst = one_data[0]
    ssh = one_data[1]
    #image = one_data[0]
    label = one_data[2]
    #convert sst, ssh, label from 1D arrays to 2D arrays
    m,n = 500,600
    Image_size = (96, 112)
    sst= tf.reshape(sst, shape = ( -1,m, n, 1), name=None)
    sst = tf.image.resize(sst, size  = Image_size, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,antialias=False, name=None)
    ssh= tf.reshape(ssh, shape = ( -1,m, n, 1), name=None)
    ssh = tf.image.resize(ssh, size  = Image_size, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,antialias=False, name=None)
    label= tf.reshape(label, shape = (-1, m, n, 1), name=None)
    image = tf.concat([sst, ssh], axis= -1, name='concat')
    
    
    if training == True:
        image , label = image_augmentation2(image, label, angles)
    

    return image, label, one_data[-2], one_data[-1]


def extract_image2(iterator):
	return iterator.get_next()


def main():

	path_tfrecord = "/home/raghavs/data/DATA_GS_train.tfrecord"
	dataset = tf.data.TFRecordDataset([path_tfrecord])
	current_time = time.strftime("%m/%d/%H/%M/%S")
	print(current_time)
	
	#for iterator
	with tf.device('/cpu:0'):
		dataset1 = dataset.map(_extract_fn,num_parallel_calls=1)
	batch_size = 32
	batch_dataset = dataset1.batch(batch_size, drop_remainder=True)
	batch_dataset= batch_dataset.shuffle(buffer_size= batch_size, seed=None, reshuffle_each_iteration=True)
	iterator = tf.compat.v1.data.make_initializable_iterator(batch_dataset)
	#iterator = dataset.make_one_shot_iterator()
	batch_data = extract_image(iterator, batch_size= batch_size, training = True)
	
	cnn= Model(batch_data, image_shape= (480, 576), sst_shape= (96, 112))
	#batch_data = extract_image2(iterator)

	config=tf.compat.v1.ConfigProto(log_device_placement=False)
	#config.gpu_options.allow_growth = True
	epoch = 100

	with tf.compat.v1.Session(config=config) as sess:
		init = tf.compat.v1.global_variables_initializer()
		sess.run(init)
		sess.run(tf.compat.v1.local_variables_initializer())

		sess.run(iterator.initializer)

		Cnn= sess.run([cnn, pad])
		

		
if __name__ == '__main__':
    #flags = read_flags()
    main()