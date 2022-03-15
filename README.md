# Facial Keypoints Detection

## Purpose

- This is an experiment with React Native (targeting iOS), Expo, and TensorFlow. It implements a simple single object detection neural network (mobilenet), a facial detection model (blazeface), plus a facial keypoint detector trained in /nn_model/. 
- Details
	- The facial training data set is from [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/) (which is in turn provided by Dr. Yoshua Bengio from the University of Montreal), and the pretrained models are from [the TensorFlow model garden](https://github.com/tensorflow/models). 
	- techniques (TBA)
	- test results (TBA)

## Installation

- Python
	1. Have python & pip installed
	2. "pip install -r requirements.txt"
	3. open ./nn_model/generate.ipynb

- React Native
	1. Have npm installed
	2. "npm install"
	3. "expo start"
	4. open in Expo Go (tested on an iPhone 12)

## Contact Info

This project is by Eric Sun - please send inquiries to [eric.sun@berkely.edu](mailto:eric.sun@berkeley.edu)

## Licensing

Copyright 2021 Eric Sun

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
