# geewhizPython

Demo of what python can do

This builds on tutorials from 

iNaturalist: https://pyinaturalist.readthedocs.io/en/stable/examples/Tutorial_1_Observations.html
TensorFlow: https://www.tensorflow.org/tutorials/images/classification

## Background

I have already created a folder with photos of [*Notophthalmus viridescens*](https://en.wikipedia.org/wiki/Eastern_newt), the Eastern newt. In the https://github.com/bomeara/geewhizR exercise on R, we found that this is the most commonly observed salamander near Knoxville, TN. It has a juvenile form that is orange and terrestrial, and a mature form that is brownish and aquatic. Machine learning is a strength of Python, so let's see if we can use it to distinguish between the two lifeforms. 

Like many things in biology, these are not discrete categories. I have used iNaturalist to get all research grade photos of *Notophthalmus viridescens* from 2022. I then manually sorted these into "Juvenile" and "Adult" folders, based on color and habitat. I removed some of the largest files. 

## Install Python

To use this, you will need to install Python3 and several libraries:

* pyinaturalist
* tensorflow
* keras
* numpy
* sklearn

And others.

## Running

This will run with `python3 salamander.py`. It will load in the images from the included folders, and then train a neural network to distinguish between the two life stages. It will then use this to predict the life stage of *Notophthalmus viridescens* from 2021 and store this in a CSV file. 

This is just a start: one could then use this to look at differences in the temporal or spatial distribution of juveniles and adults, or to look at how these have changed through time with changes in climate factors.

An important thing to remember with these data is that they are the result of individual effort: locations near roads or on busy trails are probably overrepresented, while those in remote areas are underrepresented. This is an issue with citizen science data in general. For life stages that differ in habitat, how they are noticed will often differ.