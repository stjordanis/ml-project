#######################################################################
# LIBRARY FOR LOADING LFW AND RUNNING TESTS                           #
#######################################################################

import numpy as np
import requests
import os
import sys
import tarfile
from scipy.misc import imread,imresize
from time import time

LFW_DIR = '.lfw'
NORMAL_DIR = 'lfw_normal'
FUNNELED_DIR = 'lfw_funneled'
NORMAL_URL = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
FUNNELED_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz'
PAIRS_URL = 'http://vis-www.cs.umass.edu/lfw/pairs.txt'
cache = {}

def type_to_paths(type):
	# Create the path to the file.
	if type == 'funneled':
		subdir, url = FUNNELED_DIR, FUNNELED_URL
	elif type == 'normal':
		subdir, url = NORMAL_DIR, NORMAL_URL

	return subdir, url

def load(type='funneled'):
	subdir, url = type_to_paths(type)

	# Create the folder if it doesn't exist.
	path = os.path.join('.', LFW_DIR)
	subpath = os.path.join(path, subdir)
	if not(os.path.exists(path)):
		os.makedirs(path)

	# Download the pairs file.
	if not(os.path.exists(os.path.join(path, 'pairs.txt'))):
		r = requests.get(PAIRS_URL)
		f = open(os.path.join(path, 'pairs.txt'), 'w')
		f.write(r.content)
		f.close()

	# Download the file.
	tarname = os.path.join(path, 'lfw.tgz')

	if not(os.path.exists(subpath)):
		sys.stderr.write('Downloading faces.')
		r = requests.get(url, stream=True)
		f = open(tarname, 'wb')
		for chunk in r.iter_content(chunk_size=1024):
			f.write(chunk)
		f.close()

		sys.stderr.write('Extracting faces.')
		tar = tarfile.open(tarname, 'r:gz').extractall(path=path)
		os.remove(tarname)


def get_all_images(type, resize=None, color=False):
	subdir, _ = type_to_paths(type)
	subpath = os.path.join('.', LFW_DIR, subdir)

	# Find all image files.
	images = []
	for root, _, files in os.walk(subpath):
		for file in files:
			if file.endswith('.jpg'):
				images.append(os.path.join(root, file))


	get_these_images(images, type, resize, color)

def get_these_images(images, type, resize, color):
	# Load the images.
	x0, x1, y0, y1 = 0, 250, 0, 250
	h = 250
	w = 250
	if resize is not None:
		h = int(resize * h)
		w = int(resize * w)

	# Create the numpy array for storing the images.
	if color:
		faces = np.zeros((len(images), h, w, 3), dtype=np.float32)
	else:
		faces = np.zeros((len(images), h, w), dtype=np.float32)

	# Dictionaries for mapping file numbers to names.
	num_to_name = {}
	index_to_num = {}

	for i, file in enumerate(images):
		# Read the image from the file.
		if file not in cache:
			image = imread(file)
			cache[file] = np.asarray(image, dtype=np.float32)
		face = cache[file]

		# Scale the pixel values between 0 and 1.
		face = face/255.

		# Resize the image if requested.
		if resize is not None:
			face = imresize(face, resize)

		# Eliminate colors if requested.
		if not color:
			face = face.mean(axis=2)

		faces[i, ...] = face

		# Determine the name and number of the image.
		filename = os.path.basename(file).split('.')[0]
		name, num = filename.rsplit('_', 1)
		num_to_name[int(num)] = name
		index_to_num[i] = int(num)

	return faces, index_to_num, num_to_name

def load_pairs(type, resize=None, color=False):
	load(type)
	pairs = open(os.path.join('.', LFW_DIR, 'pairs.txt')).readlines()[1:]

	# Store the output.
	sets = []

	# Create a filename for an image.
	def retrieve(name, num):
		subdir, _ = type_to_paths(type)
		filename = os.path.join('.', LFW_DIR, subdir, name, name + '_' + str(num).zfill(4) + '.jpg')
		face = get_these_images([filename], type, resize, color)[0][0]
		return (name, num, face)

	# Load.
	for set in range(10):
		new_set = []

		for match_pair in range(300):
			i = set * 600 + match_pair
			name, x, y = pairs[i].split()
			f1, f2 = retrieve(name, x), retrieve(name, y)

			new_set.append((f1, f2))

		for unmatch_pair in range(300, 600):
			i = set * 600 + unmatch_pair
			name1, x1, name2, x2 = pairs[i].split()
			f1, f2 = retrieve(name1, x1), retrieve(name2, x2)
			new_set.append((f1, f2))

		sets.append(new_set)

	return sets

def run_test_unrestricted(train, same_or_different, type, resize, color):
	sets = load_pairs(type, resize, color)
	success = 0
	false_positive = 0
	false_negative = 0
	total = 0

	# Create a holdout set.
	for test in range(len(sets)):

		# Gather the training images.
		training_images = {}
		names = {}
		for train in list(range(0, test)) + list(range(test+1, len(sets))):
			for ((name1, num1, face1), (name2, num2, face2)) in sets[train]:
				training_images[num1] = (name1, face1)
				training_images[num2] = (name2, face2)
				names[name1] = 0
				names[name2] = 0

		# Train the model.
		trained = train(training_images.values(), names.keys())

		# Test the model.
		for ((name1, num1, face1), (name2, num2, face2)) in sets[test]:
			expected = name1 == name2
			actual = same_or_different(face1, face2)
			total += 1

			if expected == actual:
				success += 1
			if expected and not(actual):
				false_negative += 1
			if actual and not(expected:
				false_positive += 1


	return {'total': total, 'success' : success, 'false_pos' : false_positive, 'false_neg' : false_negative}

