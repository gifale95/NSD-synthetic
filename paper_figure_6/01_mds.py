"""Perform MDS on the trial-average fMRI responses for the NSD-core training
images and the NSD-synthetic images.

Parameters
----------
subject : str
	String indicating the number of the subject used. If 'all', then use all
	subjects.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept vertices: only vertices above this
	threshold are used.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from sklearn.manifold import MDS
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=str, default='1')
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> MDS <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Subject loop
# =============================================================================
all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

betas = []

for s, sub in enumerate(all_subjects):


# =============================================================================
# Get indices of vertices with ncsnr above threshold
# =============================================================================
	# Load the ncsnr
	data_dir_synthetic = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+format(sub))
	metadata = np.load(os.path.join(data_dir_synthetic,
		'meatadata_nsdsynthetic.npy'), allow_pickle=True).item()
	lh_ncsnr_nsdsynthetic = metadata['lh_ncsnr']
	rh_ncsnr_nsdsynthetic = metadata['rh_ncsnr']

	# Select the above-threshold vertices
	lh_idx = np.where(lh_ncsnr_nsdsynthetic > args.ncsnr_threshold)[0]
	rh_idx = np.where(rh_ncsnr_nsdsynthetic > args.ncsnr_threshold)[0]


# =============================================================================
# Load the trial-average betas
# =============================================================================
	# NSD-core train
	data_dir_core = os.path.join(args.project_dir, 'results',
		'train_test_session_control-0', 'fmri_betas', 'zscore-'+
		str(args.zscore), 'sub-0'+format(sub))
	lh = h5py.File(os.path.join(data_dir_core,
		'lh_betas_nsdcore_train.h5'), 'r')['betas'][:,lh_idx]
	rh = h5py.File(os.path.join(data_dir_core,
		'rh_betas_nsdcore_train.h5'), 'r')['betas'][:,rh_idx]
	# Append the data from left and right hemispheres
	betas_sub_train = np.append(lh, rh, 1)
	del lh, rh

	# NSD-synthetic
	lh = h5py.File(os.path.join(data_dir_synthetic,
		'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:,lh_idx]
	rh = h5py.File(os.path.join(data_dir_synthetic,
		'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:,rh_idx]
	# Append the data from left and right hemispheres
	betas_sub_ood = np.append(lh, rh, 1)
	del lh, rh

	# Append the fMRI responses for the NSD-synthetic and NSD-core training
	# images, and append the fMRI responses across subjects
	betas.append(np.append(betas_sub_ood, betas_sub_train, 0))
	del betas_sub_train, betas_sub_ood


# =============================================================================
# Apply MDS (single subjects)
# =============================================================================
if args.subject in ['1', '2', '3', '4', '5', '6', '7', '8']:

	embedding = MDS(n_components=2, n_init=10, max_iter=1000,
		random_state=20200220)

	idx = int(args.subject) - 1
	betas_mds = embedding.fit_transform(betas[idx])


# =============================================================================
# Apply MDS (all subjects)
# =============================================================================
if args.subject == 'all':

	# Get the minimum amount of image conditions
	img_num = []
	for betas_sub in betas:
		img_num.append(len(betas_sub))
	min_img_num = min(img_num)

	# Aggregate the fMRI responses of all subjects
	for s, betas_sub in enumerate(betas):
		if s == 0:
			betas_all_sub = betas_sub[:min_img_num]
		else:
			betas_all_sub = np.append(betas_all_sub, betas_sub[:min_img_num], 1)

	# Perform MDS
	embedding = MDS(n_components=2, n_init=10, max_iter=1000,
		random_state=20200220)
	betas_mds = embedding.fit_transform(betas_all_sub)


# =============================================================================
# Save the MDS results
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'mds', 'zscore-'+str(args.zscore))

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

file_name = 'betas_mds_subject-' + args.subject + '.npy'

np.save(os.path.join(save_dir, file_name), betas_mds)
