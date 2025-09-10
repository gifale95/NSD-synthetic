"""Compute the fMRI response distance (in MDS space) between the different image
classes.

Parameters
----------
subjects : list
	List of the used NSD subjects.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept vertices: only vertices above this
	threshold are used.
project_dir : str
	Directory of the project folder.
nsd_dir : str
	Directory of the NSD.

"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()

print('>>> MDS distance scores <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the single subject MDS results
# =============================================================================
betas_mds_single_sub = []

for s, sub in enumerate(args.subjects):

	data_dir = os.path.join(args.project_dir, 'results',
		'nsdsynthetic_image_classes', 'mds', 'zscore-'+str(args.zscore),
		'betas_mds_subject-'+str(sub)+'.npy')

	betas_mds_single_sub.append(np.load(data_dir))


# =============================================================================
# Load the subject average
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'mds', 'zscore-'+str(args.zscore),
	'betas_mds_subject-all.npy')

betas_mds_all_sub = np.load(data_dir)


# =============================================================================
# Compute the euclidean distance between NSD-core train and NSD-synthetic
# responses
# =============================================================================
# Get the class indices of the NSD-synthetic images
labels_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsyntheticimageinformation.csv')
image_labels = pd.read_csv(labels_dir, sep=',')
unique_classes = list(set(image_labels['Image class']))
unique_classes.sort()

# Loop across image classes
image_class_euclidean_distance = {}
for cl in tqdm(unique_classes):

	# Get the indices of the images from the selected NSD-synthetic image class
	class_img_idx = \
		[i for i, item in enumerate(list(image_labels['Image class'])) if item == cl]

	# Loop over subjects
	distance = []
	for s, betas_sub in enumerate(betas_mds_single_sub):

		# Get the indices of the NSD-core training images
		train_img_idx = np.arange(284, len(betas_sub))

		# Compute euclidean distance between NSD-core train and NSD-synthetic
		# responses
		distance_sub = np.zeros((len(class_img_idx), len(train_img_idx)),
			dtype=np.float32)
		for i1, idx_1 in enumerate(class_img_idx):
			for i2, idx_2 in enumerate(train_img_idx):
				distance_sub[i1,i2] = np.sum(np.sqrt(np.square(
					betas_sub[idx_1] - betas_sub[idx_2])))

		# Average the euclidean distances across all images withing the same
		# NSD-synthetic image class and across all NSD-core training images
		distance_sub = np.mean(distance_sub)

		# Append the distance scores across subjects
		distance.append(distance_sub)
		del distance_sub

	# Store the distance scores for each NSD-synthetic image class
	image_class_euclidean_distance[cl] = np.asarray(distance)


# =============================================================================
# Save the MDS results
# =============================================================================
results = {
	'betas_mds_single_sub': betas_mds_single_sub,
	'betas_mds_all_sub': betas_mds_all_sub,
	'image_class_euclidean_distance': image_class_euclidean_distance
	}

save_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'mds', 'zscore-'+str(args.zscore))

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

np.save(os.path.join(save_dir, 'mds.npy'), results)
