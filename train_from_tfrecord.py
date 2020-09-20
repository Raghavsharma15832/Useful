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
        'iamge': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
       
    }

    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, features)

    sst = tf.io.decode_raw(sample['image'], tf.float64)     
    img_shape =sample["Image shapes"]
    label = tf.io.decode_raw(sample['label'], tf.uint8)
    date = sample['filename']
    
    

    return [image label, img_shape, date]        

def image_augmentation2(image, label, angles= 0):
    image = tfa.image.rotate(images= image,
                angles= angles ,interpolation= 'BILINEAR',name= None)
    label = tfa.image.rotate(images= label,
                angles=  angles,interpolation= 'BILINEAR',name= None)
    return image, label

def extract_image(iterator, batch_size, angles= 0, training= False):

    one_data = iterator.get_next()
    '''
    one_data is a list of tensor contains [image,  label, img_shape, filename]
    The function returns a list of [augmentated image, label, image_shape, filename]
    '''
    image_shape = one_data[3]
    image = one_data[0] # this will be a 1D array
    label = one_data[1] # this will be a 1 D array

    #convert image,  label from 1D arrays to 2D arrays
    m,n = image_shape
    image= tf.reshape(image, shape = ( -1,m, n, 1), name=None) # If working with RGB image put 3 instread of 1
    sst = tf.image.resize(image, size  = Image_size, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,antialias=False, name=None)
    label= tf.reshape(label, shape = (-1, m, n, 1), name=None)
    
    #If you want to augment the images ( here augmentation has been performed by rotating images with some angle )
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