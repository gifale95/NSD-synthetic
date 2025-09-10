"""Test the encoding model's prediction accuracy independently for each
NSD-synthetic image class.

Parameters
----------
subjects : list
	List of the used NSD subjects.
nsdsynthetic_image_class : int
	Integer indicating the NSD-synthetic image class used.
	0 for chromatic noise.
	1 for contrast modulation.
	2 for manipulated scenes.
	3 for natural scnees.
	4 for noise.
	5 for phase-coherence modulation.
	6 for single words.
	7 for spiral gratings.
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
import h5py
from scipy.stats import pearsonr
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--nsdsynthetic_image_class', type=int, default=0)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()

print('>>> Test encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get the indices of the images from the selected NSD-synthetic image class
# =============================================================================
# Load the used NSD-synthetic image classses
labels_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsyntheticimageinformation.csv')
image_labels = pd.read_csv(labels_dir, sep=',')
unique_classes = list(set(image_labels['Image class']))
unique_classes.sort()

# Get the indices of the images from the selected image class
class_img_idx = \
	[i for i, item in enumerate(list(image_labels['Image class'])) if item == unique_classes[args.nsdsynthetic_image_class]]


# =============================================================================
# Load the recorded and predicted fMRI responses for the NSD-synthetic images
# =============================================================================
# Predicted fMRI
data_dir = os.path.join(args.project_dir, 'results',
	'train_test_session_control-0', 'predicted_fmri', 'zscore-'+
	str(args.zscore), 'model-'+args.model, 'layer-all', 'regression-linear',
	'predicted_fmri_sub-0'+str(args.subject)+'.npy')
data = np.load(data_dir, allow_pickle=True).item()
lh_betas_pred = data['lh_betas_nsdsynthetic_pred'][class_img_idx].astype(np.float32)
rh_betas_pred = data['rh_betas_nsdsynthetic_pred'][class_img_idx].astype(np.float32)
del data

# Recorded fMRI
data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
	'zscore-'+str(args.zscore), 'sub-0'+str(args.subject))
lh_betas = h5py.File(os.path.join(data_dir,
	'lh_betas_nsdsynthetic.h5'), 'r')['betas'][class_img_idx].astype(np.float32)
rh_betas = h5py.File(os.path.join(data_dir,
	'rh_betas_nsdsynthetic.h5'), 'r')['betas'][class_img_idx].astype(np.float32)

# ncsnr
metadata = np.load(os.path.join(data_dir, 'meatadata_nsdsynthetic.npy'),
	allow_pickle=True).item()
lh_ncsnr_nsdsynthetic = metadata['lh_ncsnr']
rh_ncsnr_nsdsynthetic = metadata['rh_ncsnr']

# Convert the ncsnr to noise ceiling
img_reps_2 = 236
img_reps_4 = 32
img_reps_8 = 8
img_reps_10 = 8
norm_term = (img_reps_2/2 + img_reps_4/4 + img_reps_8/8 + img_reps_10/10) / \
	(img_reps_2 + img_reps_4 + img_reps_8 + img_reps_10)
lh_nc_nsdsynthetic = (lh_ncsnr_nsdsynthetic ** 2) / \
	((lh_ncsnr_nsdsynthetic ** 2) + norm_term).astype(np.float32)
rh_nc_nsdsynthetic = (rh_ncsnr_nsdsynthetic ** 2) / \
	((rh_ncsnr_nsdsynthetic ** 2) + norm_term).astype(np.float32)


# =============================================================================
# Compute the noise ceiling for the chosen NSD-synthetic image class
# =============================================================================
# Load the ncsnr
lh_nc = metadata['lh_ncsnr_classes'][unique_classes[args.nsdsynthetic_image_class]]
rh_nc = metadata['rh_ncsnr_classes'][unique_classes[args.nsdsynthetic_image_class]]

# Get the image repeats
img_reps_2 = 0
img_reps_4 = 0
img_reps_8 = 0
img_reps_10 = 0
img_repeats = metadata['nsdsynthetic_img_repeats']
for i, img in enumerate(class_img_idx):
	if img_repeats[img] == 2:
		img_reps_2 += 1
	elif img_repeats[img] == 4:
		img_reps_4 += 1
	elif img_repeats[img] == 8:
		img_reps_8 += 1
	elif img_repeats[img] == 10:
		img_reps_10 += 1

# Convert the ncsnr to noise ceiling
norm_term = (img_reps_2/2 + img_reps_4/4 + img_reps_8/8 + img_reps_10/10) / \
	(img_reps_2 + img_reps_4 + img_reps_8 + img_reps_10)
lh_nc = (lh_nc ** 2) / ((lh_nc ** 2) + norm_term).astype(np.float32)
rh_nc = (rh_nc ** 2) / ((rh_nc ** 2) + norm_term).astype(np.float32)

# Add a very small number to noise ceiling values of 0, otherwise the
# noise-ceiling-normalized encoding accuracy cannot be calculated (division by 0
# is not possible)
lh_nc[lh_nc==0] = 1e-14
rh_nc[rh_nc==0] = 1e-14


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
# Compute the correlation between predicted and recorded fMRI responses
lh_correlation = np.zeros(lh_betas_pred.shape[1], dtype=np.float32)
rh_correlation = np.zeros(rh_betas_pred.shape[1], dtype=np.float32)
for v in range(lh_betas_pred.shape[1]):
	lh_correlation[v] = pearsonr(lh_betas_pred[:,v], lh_betas[:,v])[0]
	rh_correlation[v] = pearsonr(rh_betas_pred[:,v], rh_betas[:,v])[0]

# Set negative correlation scores to zero
lh_correlation[lh_correlation<0] = 0
rh_correlation[rh_correlation<0] = 0

# Turn the correlations into r2 scores
lh_r2 = lh_correlation ** 2
rh_r2 = rh_correlation ** 2

# Compute the noise-ceiling-normalized encoding accuracy
lh_explained_variance = np.divide(lh_r2, lh_nc) * 100
rh_explained_variance = np.divide(rh_r2, rh_nc) * 100

# Set the noise-ceiling-normalized encoding accuracy to 100 for vertices
# where the the correlation is higher than the noise ceiling, to prevent
# encoding accuracy values higher than 100%
lh_explained_variance[lh_explained_variance>100] = 100
rh_explained_variance[rh_explained_variance>100] = 100


# =============================================================================
# Save the prediction accuracy
# =============================================================================
results = {
	'lh_correlation': lh_correlation,
	'rh_correlation': rh_correlation,
	'lh_r2': lh_r2,
	'rh_r2': rh_r2,
	'lh_nc': lh_nc,
	'rh_nc': rh_nc,
	'lh_explained_variance' : lh_explained_variance,
	'rh_explained_variance': rh_explained_variance,
	'lh_nc_all_nsdsynthetic_image_classes': lh_nc_nsdsynthetic,
	'rh_nc_all_nsdsynthetic_image_classes': rh_nc_nsdsynthetic
	}

save_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'encoding_accuracy', 'zscore-'+
	str(args.zscore), 'model-'+args.model)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy_nsdsynthetic_sub-' + \
	format(args.subject, '02') + '_image_class-' + \
	str(args.nsdsynthetic_image_class) + '.npy'

np.save(os.path.join(save_dir, file_name), results)
