"""Perform multidimensional scaling (MDS) jointly on NSD-synthetic's fMRI
responses of all subjects and of all vertices with ncsnr above a threshold.

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

"""

import argparse
import os
import numpy as np
import h5py
from sklearn.manifold import MDS

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> MDS | NSD-synthetic <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Subjects loop
# =============================================================================
retained_vertices = {}

for s, sub in enumerate(args.subjects):


# =============================================================================
# Load the ncsnr
# =============================================================================
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+format(sub))
	metadata = np.load(os.path.join(data_dir, 'meatadata_nsdsynthetic.npy'),
		allow_pickle=True).item()

	# ncsnr
	lh_ncsnr = metadata['lh_ncsnr']
	rh_ncsnr = metadata['rh_ncsnr']


# =============================================================================
# Get indices of vertices with ncsnr above threshold
# =============================================================================
	lh_idx = np.where(lh_ncsnr > args.ncsnr_threshold)[0]
	rh_idx = np.where(rh_ncsnr > args.ncsnr_threshold)[0]


# =============================================================================
# Load the NSD-synthetic data
# =============================================================================
	lh_betas = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:,lh_idx]
	rh_betas = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:,rh_idx]

	# Append the data from left and right hemispheres
	betas_subject = np.append(lh_betas, rh_betas, 1)
	del lh_betas, rh_betas

	# Convert NaN values (missing data) to zeros
	betas_subject = np.nan_to_num(betas_subject)

	# Store the number of retained vertices
	retained_vertices['s'+str(sub)] = betas_subject.shape[1]

	# Append the data across subjects
	if s == 0:
		betas = betas_subject
	else:
		betas = np.append(betas, betas_subject, 1)
	del betas_subject


# =============================================================================
# Perform MDS
# =============================================================================
embedding = MDS(n_components=2, n_init=10, max_iter=1000,
	random_state=20200220)

betas_mds = embedding.fit_transform(betas)


# =============================================================================
# Save the MDS results
# =============================================================================
results = {
	'betas_mds': betas_mds,
	'retained_vertices': retained_vertices
	}

save_dir = os.path.join(args.project_dir, 'results', 'mds_nsdsynthetic',
	'zscore-'+str(args.zscore))
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'mds_nsdsynthetic_ncsnr_threshold-' + \
	format(args.ncsnr_threshold, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), results)
