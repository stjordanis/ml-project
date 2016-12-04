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
from multiprocessing import Process, Queue

LFW_DIR = '.lfw'
NORMAL_DIR = 'lfw_normal'
FUNNELED_DIR = 'lfw_funneled'
NORMAL_URL = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
FUNNELED_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz'
PAIRS_URL = 'http://vis-www.cs.umass.edu/lfw/pairs.txt'
cache = {}

# Convert a type into the appropriate data path.
def type_to_paths(type):
	# Create the path to the file.
	if type == 'funneled':
		subdir, url = FUNNELED_DIR, FUNNELED_URL
	elif type == 'normal':
		subdir, url = NORMAL_DIR, NORMAL_URL

	return subdir, url

# Download images.
def download(type='funneled'):
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

# Retrieve all images from disk.
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

# Retrieve a specific set of images from disk (provided via filename).
def get_these_images(images, type, resize, color, crop):
	# Load the images.
	x0, x1, y0, y1 = 0, 250, 0, 250

	if crop is not None:
		ww, hh = crop
		offw, offh = (x1 - x0 - ww)//2, (y1 - y0 - hh)//2
		x0 += offw
		y0 += offh
		x1 -= offw
		y1 -= offh

	w = x1 - x0
	h = y1 - y0

	if resize is not None:
		h = int(resize * h)
		w = int(resize * w)

	# Create the numpy array for storing the images.
	if color:
		faces = np.zeros((len(images), w, h, 3), dtype=np.float64)
	else:
		faces = np.zeros((len(images), w, h), dtype=np.float64)

	# Load the images.
	for i, file in enumerate(images):
		# Read the image from the file.
		if os.path.basename(file) not in cache:
			image = imread(file)

			# Cache and scale the image.
			cache[os.path.basename(file)] = np.asarray(image, dtype=np.float64) / 255.
		face = cache[os.path.basename(file)]

		# Resize the image if requested.
		face = face[x0:x1][y0:y1]
		if resize is not None:
			face = imresize(face, resize)

		# Eliminate colors if requested.
		if not color:
			face = face.mean(axis=2)

		faces[i, ...] = face

		# Determine the name and number of the image.
		filename = os.path.basename(file).split('.')[0]

	return faces

def load_pairs(type, resize=None, folds=10, color=False, crop=None):
	download(type)
	pairs = open(os.path.join('.', LFW_DIR, 'pairs.txt')).readlines()[1:]

	# Store the output.
	sets = []
	retrieved = {}
	# Create a filename for an image.
	def retrieve(name, num):
		# Get from cache.
		if name + str(num) in retrieved:
			return (name, num, retrieved[name + str(num)])

		# Otherwise, retrieve from disk and cache.
		subdir, _ = type_to_paths(type)
		filename = os.path.join('.', LFW_DIR, subdir, name, name + '_' + str(num).zfill(4) + '.jpg')
		face = get_these_images([filename], type, resize, color, crop)[0]
		retrieved[name + str(num)] = face
		return (name, num, face)

	# Load.
	for set in range(folds):
		new_set = []

		# Extract match pairs.
		for match_pair in range(300):
			i = set * 600 + match_pair
			name, x, y = pairs[i].split()
			f1, f2 = retrieve(name, x), retrieve(name, y)

			new_set.append((f1, f2))

		# Extract non-match pairs.
		for unmatch_pair in range(300, 600):
			i = set * 600 + unmatch_pair
			name1, x1, name2, x2 = pairs[i].split()
			f1, f2 = retrieve(name1, x1), retrieve(name2, x2)
			new_set.append((f1, f2))

		sets.append(new_set)

	return sets

# Pairwise determines whether we run a verification regime or n individual classification
# regime. If pairwise == True, then outcome_fn takes two faces and returns True if they are
# the same person and False otherwise.
# If pairwise == False, then outcome_fn takes one face and returns the name associated with
# it (or None if it doesn't belong to any trained identity).
def run_test(folds, train_fn, outcome_fn, type, resize, color, crop=None):
	sets = load_pairs(type, resize=resize, folds=10, color=color, crop=crop)
	print(len(sets[0]))
	success = 0
	false_positive = 0
	true_positive = 0
	true_negative = 0
	false_negative = 0
	total = 0

	# Create a holdout set.
	for test in range(folds):

		# Gather the training images.
		names = []
		pairs = []
		targets = []
		for j in list(range(0, test)) + list(range(test+1, len(sets))):
			for ((name1, num1, face1), (name2, num2, face2)) in sets[j]:
				names.append((name1, name2))
				pairs.append((face1, face2))
				targets.append(name1 == name2)

		# Train the model.
		trained = train_fn(pairs, targets, names)

		# Test the model.
		for ((name1, num1, face1), (name2, num2, face2)) in sets[test]:
				expected = name1 == name2
				actual = outcome_fn(face1, face2, trained)
				total += 1

				if expected == actual and expected:
					true_positive += 1
				if expected == actual and not expected:
					true_negative += 1
				if expected and not(actual):
					false_negative += 1
				if actual and not(expected):
					false_positive += 1

	return {'total': total, 'true_pos' : true_positive, 'true_neg' : true_negative, 'false_pos' : false_positive, 'false_neg' : false_negative}
