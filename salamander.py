from pyinaturalist import *
import os
import requests # request img from web
import shutil # save img locally
import urllib.request # request img from web
import keras
import sklearn
from sklearn.metrics import classification_report,confusion_matrix
import pathlib
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

new_dir = "salamander_images_2022"

# Manually sorted into Juvenile and Adult folders

# From https://www.tensorflow.org/tutorials/images/classification

data_dir = pathlib.Path(new_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
class_names = train_ds.class_names
print(class_names)


num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

print("Model trained")


print("Now getting new observations to test")

focal_year = '2021'

response = get_taxa(q='Notophthalmus viridescens', rank=['species'])

taxa = Taxon.from_json_list(response)

response = get_observations(taxon_id=taxa[0].id, page='all', quality_grade='research', year=focal_year, photo_license='CC-BY', photos='true')

salamander_observations = Observation.from_json_list(response)

pprint(salamander_observations[:20])

#print(salamander_observations[0])

df = pd.DataFrame(columns = ["ID", "Latitude", "Longitude","Observed_on", "Stage", "Stage_confidence", "Photo_URL"])

def appendDictToDF(df,dictToAppend):
  df = pd.concat([df, pd.DataFrame.from_records([dictToAppend])])
  return df

for obs in salamander_observations[:50]:
	try:
		latitude = obs.location[0]
		longitude = obs.location[1]
		ID = obs.id
		observed_on = obs.observed_on
		photo_url = obs.photo_url
		print(photo_url)
		sample_path = tf.keras.utils.get_file("photo_" + str(ID), origin=photo_url)
		img = tf.keras.utils.load_img(sample_path, target_size=(img_height, img_width))
		img_array = tf.keras.utils.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0) # Create a batch
		predictions = model.predict(img_array)
		score = tf.nn.softmax(predictions[0])
		stage = class_names[np.argmax(score)]
		stage_confidence = 100 * np.max(score)
		print(stage)
		print(stage_confidence)
		df = appendDictToDF(df,{"ID": ID, "Latitude": latitude, "Longitude": longitude, "Observed_on": observed_on, "Stage": stage, "Stage_confidence": stage_confidence, "Photo_URL": photo_url})
	except:
		print("Error in observation " + str(ID))
		pass

df.to_csv("salamander_observations_2021.csv")