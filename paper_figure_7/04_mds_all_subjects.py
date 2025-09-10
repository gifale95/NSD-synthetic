"""Perform MDS on the trial-average fMRI responses for the NSD-core and
NSD-synthetic images.

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
	# Load the ncsnr (NSD-core)
	data_dir_core = os.path.join(args.project_dir, 'results',
		'nsdcore_id_ood_tests', 'fmri_betas', 'zscore-'+
		str(args.zscore), 'sub-0'+format(sub))
	metadata = np.load(os.path.join(data_dir_core, 'meatadata_nsdcore.npy'),
		allow_pickle=True).item()
	lh_idx_core = metadata['lh_ncsnr'] > args.ncsnr_threshold
	rh_idx_core = metadata['rh_ncsnr'] > args.ncsnr_threshold

	# Load the ncsnr (NSD-synthetic)
	data_dir_synthetic = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+format(sub))
	metadata = np.load(os.path.join(data_dir_synthetic,
		'meatadata_nsdsynthetic.npy'), allow_pickle=True).item()
	lh_idx_nsdsynthetic = metadata['lh_ncsnr'] > args.ncsnr_threshold
	rh_idx_nsdsynthetic = metadata['rh_ncsnr'] > args.ncsnr_threshold

	# Select the above-threshold vertices
	lh_idx = np.where(np.logical_and(lh_idx_core, lh_idx_nsdsynthetic))[0]
	rh_idx = np.where(np.logical_and(rh_idx_core, rh_idx_nsdsynthetic))[0]


# =============================================================================
# Load the trial-average betas
# =============================================================================
	# NSD-core train
	lh = h5py.File(os.path.join(data_dir_core,
		'lh_betas_nsdcore_train.h5'), 'r')['betas'][:,lh_idx]
	rh = h5py.File(os.path.join(data_dir_core,
		'rh_betas_nsdcore_train.h5'), 'r')['betas'][:,rh_idx]
	# Append the data from left and right hemispheres
	betas_sub_train = np.append(lh, rh, 1)
	del lh, rh

	# NSD-core ID testing
	lh = h5py.File(os.path.join(data_dir_core,
		'lh_betas_nsdcore_test_id.h5'), 'r')['betas'][:,lh_idx]
	rh = h5py.File(os.path.join(data_dir_core,
		'rh_betas_nsdcore_test_id.h5'), 'r')['betas'][:,rh_idx]
	# Append the data from left and right hemispheres
	betas_sub_core_test_id = np.append(lh, rh, 1)
	del lh, rh

	# NSD-core OOD testing
	lh = h5py.File(os.path.join(data_dir_core,
		'lh_betas_nsdcore_test_ood.h5'), 'r')['betas'][:,lh_idx]
	rh = h5py.File(os.path.join(data_dir_core,
		'rh_betas_nsdcore_test_ood.h5'), 'r')['betas'][:,rh_idx]
	# Append the data from left and right hemispheres
	betas_sub_core_test_ood = np.append(lh, rh, 1)
	del lh, rh

	# NSD-synthetic
	lh = h5py.File(os.path.join(data_dir_synthetic,
		'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:,lh_idx]
	rh = h5py.File(os.path.join(data_dir_synthetic,
		'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:,rh_idx]
	# Append the data from left and right hemispheres
	betas_sub_synthetic = np.append(lh, rh, 1)
	del lh, rh

	# Append the the fMRI responses for the NSD-synthetic, NSD-core ID/OOD
	# testing images, and NSD-core trainig images, and append the fMRI responses
	# across subjects
	betas_sub = np.append(betas_sub_core_test_id, betas_sub_core_test_ood, 0)
	betas_sub = np.append(betas_sub, betas_sub_synthetic, 0)
	betas_sub = np.append(betas_sub, betas_sub_train, 0)
	betas.append(betas_sub)
	del betas_sub_train, betas_sub_core_test_id, betas_sub_core_test_ood, \
		betas_sub_synthetic, betas_sub


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
	'nsdcore_id_ood_tests', 'mds_all_subjects', 'zscore-'+str(args.zscore))

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

file_name = 'betas_mds_subject-' + args.subject + '.npy'

np.save(os.path.join(save_dir, file_name), betas_mds)
