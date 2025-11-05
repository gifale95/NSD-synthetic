"""Test the image condition identification accuracy of the encoding model's in
silico neural responses for NSD-synthetic's images.

Parameters
----------
subjects : list
	List of the used NSD subjects.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept vertices: only vertices above this
	threshold are used.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet', 'resnet50', 'moco', and 'vit_b_32'.
project_dir : str
	Directory of the project folder.
nsd_dir : str
	Directory of the NSD.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import h5py
from scipy.stats import pearsonr
import pandas as pd
from sklearn.utils import resample
import random

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()

print('>>> Image condition identification <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Load the NSD-synthetic image classes
# =============================================================================
labels_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsyntheticimageinformation.csv')
image_labels = pd.read_csv(labels_dir, sep=',')
unique_classes = list(set(image_labels['Image class']))
unique_classes.sort()


# =============================================================================
# Empty result lists
# =============================================================================
correlation_matrix = []
rank_nsdsynthetic = []


# =============================================================================
# Loop across subjects
# =============================================================================
for s, sub in enumerate(tqdm(args.subjects)):


# =============================================================================
# Get indices of vertices with ncsnr above threshold
# =============================================================================
	# NSD-synthetic
	metadata_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+format(sub),
		'meatadata_nsdsynthetic.npy')
	metadata = np.load(metadata_dir, allow_pickle=True).item()
	lh_ncsnr_nsdsynthetic = metadata['lh_ncsnr']
	rh_ncsnr_nsdsynthetic = metadata['rh_ncsnr']
	lh_idx = np.where(lh_ncsnr_nsdsynthetic > args.ncsnr_threshold)[0]
	rh_idx = np.where(rh_ncsnr_nsdsynthetic > args.ncsnr_threshold)[0]


# =============================================================================
# Load the recorded and predicted fMRI responses for the NSD-synthetic images
# =============================================================================
	# Recorded fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+str(sub))
	lh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:,lh_idx].astype(np.float32)
	rh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:,rh_idx].astype(np.float32)

	# Predicted fMRI
	data_dir = os.path.join(args.project_dir, 'results',
		'train_test_session_control-0', 'predicted_fmri', 'zscore-'+
		str(args.zscore), 'model-'+args.model, 'layer-all', 'regression-linear',
		'predicted_fmri_sub-0'+str(sub)+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	lh_betas_nsdsynthetic_pred = data['lh_betas_nsdsynthetic_pred'][:,lh_idx].astype(np.float32)
	rh_betas_nsdsynthetic_pred = data['rh_betas_nsdsynthetic_pred'][:,rh_idx].astype(np.float32)
	del data

	# Append the fMRI responses across vertices
	betas_nsdsynthetic = np.append(lh_betas_nsdsynthetic,
		rh_betas_nsdsynthetic, 1)
	betas_nsdsynthetic_pred = np.append(lh_betas_nsdsynthetic_pred,
		rh_betas_nsdsynthetic_pred, 1)
	del lh_betas_nsdsynthetic, rh_betas_nsdsynthetic, \
		lh_betas_nsdsynthetic_pred, rh_betas_nsdsynthetic_pred, \


# =============================================================================
# Correlate the recorded and predicted fMRI responses of each image condition
# =============================================================================
	correlation_matrix_sub = np.zeros((len(betas_nsdsynthetic),
		len(betas_nsdsynthetic)), dtype=np.float32)

	for i1 in range(len(betas_nsdsynthetic)):
		for i2 in range(len(betas_nsdsynthetic)):
			correlation_matrix_sub[i1,i2] = pearsonr(betas_nsdsynthetic[i1],
				betas_nsdsynthetic_pred[i2])[0]

	correlation_matrix.append(correlation_matrix_sub)


# =============================================================================
# Get the identification rank of each image condition
# =============================================================================
# To mitigate the risk of the identification results reflecting imbalances
# between the number of images in each NSD-synthetic image class, each image
# is ranked based on its correlation scores with 8 images from each image class
# (since every class has at least 8 images), for a total of 64 images. The
# identification algorithm is iterated N times, while each time computing the
# rankings using the correlations with a different random sample of 8 images per
# class. The final identification accuracies consist of the average across these
# N iterations.

	# Identification parameters and empty results variable
	min_img = 8
	n_iter = 1000
	rank_nsdsynthetic_sub = np.zeros((len(correlation_matrix_sub), n_iter),
		dtype=np.int32)

	# Get image class indices
	class_img_idx = {}
	for cl in unique_classes:
		class_img_num = \
			[i for i, item in enumerate(list(image_labels['Image class'])) if item == cl]
		class_img_idx[cl] = np.asarray(class_img_num)

	# Loop across images
	for i in tqdm(range(len(correlation_matrix_sub))):

		# Get the correlations scores of the target image
		correlations = correlation_matrix_sub[i]

		# Loop across iterations
		for it in range(n_iter):

			# Randomly select the correlations of 8 images from each image
			# class, while ensuring that the target image is within the 8 images
			# of the corresponding image class
			candidate_img = np.empty(0).astype(np.float32)
			for cl in unique_classes:
				data = correlations[class_img_idx[cl]]
				# If the target image is from the current image class
				if i in class_img_idx[cl]:
					idx_target = np.where(class_img_idx[cl] == i)[0][0]
					data = np.delete(data, idx_target)
					n_samples = min(min_img, len(data))
					corr = resample(data, replace=False, n_samples=n_samples)
					# Add the target image
					if n_samples < min_img:
						corr = np.insert(corr, 0, correlations[i])
					else:
						corr[0] = correlations[i]
					idx_correct = len(candidate_img)
				# If the target image is not from the current image class
				else:
					corr = resample(data, replace=False, n_samples=min_img)
				candidate_img = np.append(candidate_img, corr)

			# Get the identification rank of each image condition
			rank_nsdsynthetic_sub[i] = np.where(np.argsort(
				candidate_img)[::-1] == idx_correct)[0][0]

	del correlation_matrix_sub

	# Store the identification ranks (averaged across the N iterations)
	rank_nsdsynthetic.append(np.mean(rank_nsdsynthetic_sub, 1))


# =============================================================================
# Format the result lists into numpy arrays
# =============================================================================
correlation_matrix = np.asarray(correlation_matrix)
rank_nsdsynthetic = np.asarray(rank_nsdsynthetic)


# =============================================================================
# Aggregate the identification ranks across image classes
# =============================================================================
scores_rank_nsdsynthetic_classes = {}
mean_rank_nsdsynthetic_classes = {}

for cl in unique_classes:

	# Select images from a given image class
	class_img_num = \
		[i for i, item in enumerate(list(image_labels['Image class'])) if item == cl]
	class_img_num = np.asarray(class_img_num)
	class_ranks = rank_nsdsynthetic[:,class_img_num].flatten()

	# Store the ranks for each image class
	scores_rank_nsdsynthetic_classes[cl] = class_ranks
	mean_rank_nsdsynthetic_classes[cl] = np.mean(class_ranks)
	del class_ranks


# =============================================================================
# Save the prediction accuracy
# =============================================================================
results = {
	'correlation_matrix': correlation_matrix,
	'rank_nsdsynthetic': rank_nsdsynthetic,
	'scores_rank_nsdsynthetic_classes': scores_rank_nsdsynthetic_classes,
	'mean_rank_nsdsynthetic_classes': mean_rank_nsdsynthetic_classes
	}

save_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'image_identification_accuracy', 'zscore-'+
	str(args.zscore), 'model-'+args.model)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'image_identification_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)
