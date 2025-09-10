"""Perform MDS on the trial-average fMRI responses for the NSD-core images.

Parameters
----------
subject : int
	Number of the used subject.
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
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> MDS <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get indices of vertices with ncsnr above threshold
# =============================================================================
# Load the ncsnr
data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+format(args.subject))
metadata = np.load(os.path.join(data_dir, 'meatadata_nsdcore.npy'),
	allow_pickle=True).item()
lh_ncsnr = metadata['lh_ncsnr']
rh_ncsnr = metadata['rh_ncsnr']

# Select the above-threshold vertices
lh_idx = np.where(lh_ncsnr > args.ncsnr_threshold)[0]
rh_idx = np.where(rh_ncsnr > args.ncsnr_threshold)[0]


# =============================================================================
# Load the trial-average betas
# =============================================================================
# NSD-core betas
lh = h5py.File(os.path.join(data_dir, 'lh_betas_nsdcore.h5'),
	'r')['betas'][:,lh_idx]
rh = h5py.File(os.path.join(data_dir, 'rh_betas_nsdcore.h5'),
	'r')['betas'][:,rh_idx]
# Append the data from left and right hemispheres
betas = np.append(lh, rh, 1)
del lh, rh

# Image numbers
img_num = metadata['img_num']


# =============================================================================
# Apply MDS
# =============================================================================
embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=20200220)

betas_mds = embedding.fit_transform(betas)


# =============================================================================
# Save the MDS results
# =============================================================================
results = {
	'img_num': img_num,
	'betas_mds': betas_mds
	}

save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'mds_single_subjects', 'zscore-'+str(args.zscore))

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

file_name = 'betas_mds_subject-' + format(args.subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), results)
