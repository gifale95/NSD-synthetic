"""Perform multidimensional scaling (MDS) on NSD-synthetic's and NSD-core's
(scan sessions 10 and 20) fMRI responses of all subjects and of all vertices
with ncsnr above a threshold.

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
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import nibabel as nib
from scipy.stats import zscore
from sklearn.manifold import MDS

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()

print('>>> MDS | NSD-synthetic & NSD-core <<<')
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
	# NSD-synthetic
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+format(sub))
	metadata = np.load(os.path.join(data_dir, 'meatadata_nsdsynthetic.npy'),
		allow_pickle=True).item()
	lh_ncsnr_nsdsynthetic = metadata['lh_ncsnr']
	rh_ncsnr_nsdsynthetic = metadata['rh_ncsnr']

	# NSD-core
	data_dir = os.path.join(args.project_dir, 'results',
		'train_test_session_control-0', 'fmri_betas', 'zscore-'+
		str(args.zscore), 'sub-0'+format(sub))
	metadata = np.load(os.path.join(data_dir, 'meatadata_nsdcore.npy'),
		allow_pickle=True).item()
	lh_ncsnr_nsdcore = metadata['lh_ncsnr']
	rh_ncsnr_nsdcore = metadata['rh_ncsnr']


# =============================================================================
# Get indices of vertices with ncsnr above threshold
# =============================================================================
	# NSD-synthetic
	lh_idx_nsdsynthetic = lh_ncsnr_nsdsynthetic > args.ncsnr_threshold
	rh_idx_nsdsynthetic = rh_ncsnr_nsdsynthetic > args.ncsnr_threshold

	# NSD-core
	lh_idx_nsdcore = lh_ncsnr_nsdcore > args.ncsnr_threshold
	rh_idx_nsdcore = rh_ncsnr_nsdcore > args.ncsnr_threshold

	# Only retain vertices with nscnr above threshold for both NSD-synthetic
	# and NSD-core
	lh_idx = np.where(np.logical_and(lh_idx_nsdsynthetic, lh_idx_nsdcore))[0]
	rh_idx = np.where(np.logical_and(rh_idx_nsdsynthetic, rh_idx_nsdcore))[0]

	# Store the number of retained vertices
	retained_vertices['s'+str(sub)] = len(lh_idx) + len(rh_idx)


# =============================================================================
# Load the fMRI responses
# =============================================================================
	# NSD-synthetic
	betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
		format(sub, '02'), 'fsaverage',
		'nsdsyntheticbetas_fithrf_GLMdenoise_RR')
	# Load the fMRI betas
	lh_file_name = 'lh.betas_nsdsynthetic.mgh'
	rh_file_name = 'rh.betas_nsdsynthetic.mgh'
	lh_betas = np.transpose(np.squeeze(nib.load(os.path.join(
		betas_dir, lh_file_name)).get_fdata())).astype(np.float32)[:,lh_idx]
	rh_betas = np.transpose(np.squeeze(nib.load(os.path.join(
		betas_dir, rh_file_name)).get_fdata())).astype(np.float32)[:,rh_idx]
	# Append the data from left and right hemispheres
	betas_subject = np.append(lh_betas, rh_betas, 1)
	del lh_betas, rh_betas
	# z-score the betas of each vertex within the scan session
	if args.zscore == 1:
		betas_subject = zscore(betas_subject, nan_policy='omit')
	# Append the data across subjects
	if s == 0:
		betas_nsdsynthetic = betas_subject
	else:
		betas_nsdsynthetic = np.append(betas_nsdsynthetic, betas_subject, 1)
	del betas_subject

	# NSD-core (session 10)
	betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
		format(sub, '02'), 'fsaverage',
		'betas_fithrf_GLMdenoise_RR')
	# Load the fMRI betas
	lh_file_name = 'lh.betas_session10.mgh'
	rh_file_name = 'rh.betas_session10.mgh'
	lh_betas = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		lh_file_name)).get_fdata())).astype(np.float32)[:,lh_idx]
	rh_betas = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		rh_file_name)).get_fdata())).astype(np.float32)[:,rh_idx]
	# Match trial number with NSD-synthetic
	lh_betas = lh_betas[:744]
	rh_betas = rh_betas[:744]
	# Append the data from left and right hemispheres
	betas_subject = np.append(lh_betas, rh_betas, 1)
	del lh_betas, rh_betas
	# z-score the betas of each vertex within the scan session
	if args.zscore == 1:
		betas_subject = zscore(betas_subject, nan_policy='omit')
	# Append the data across subjects
	if s == 0:
		betas_nsdcore_sess_1 = betas_subject
	else:
		betas_nsdcore_sess_1 = np.append(betas_nsdcore_sess_1, betas_subject, 1)
	del betas_subject

	# NSD-core (session 20)
	betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
		format(sub, '02'), 'fsaverage',
		'betas_fithrf_GLMdenoise_RR')
	# Load the fMRI betas
	lh_file_name = 'lh.betas_session20.mgh'
	rh_file_name = 'rh.betas_session20.mgh'
	lh_betas = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		lh_file_name)).get_fdata())).astype(np.float32)[:,lh_idx]
	rh_betas = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		rh_file_name)).get_fdata())).astype(np.float32)[:,rh_idx]
	# Match trial number with NSD-synthetic
	lh_betas = lh_betas[:744]
	rh_betas = rh_betas[:744]
	# Append the data from left and right hemispheres
	betas_subject = np.append(lh_betas, rh_betas, 1)
	del lh_betas, rh_betas
	# z-score the betas of each vertex within the scan session
	if args.zscore == 1:
		betas_subject = zscore(betas_subject, nan_policy='omit')
	# Append the data across subjects
	if s == 0:
		betas_nsdcore_sess_2 = betas_subject
	else:
		betas_nsdcore_sess_2 = np.append(betas_nsdcore_sess_2, betas_subject, 1)
	del betas_subject


# =============================================================================
# Append the NSD-synthetic and NSD-core data across trials
# =============================================================================
betas = np.append(betas_nsdsynthetic, betas_nsdcore_sess_1, 0)
betas = np.append(betas, betas_nsdcore_sess_2, 0)
del betas_nsdsynthetic, betas_nsdcore_sess_1, betas_nsdcore_sess_2

# Convert NaN values (missing data) to zeros
betas = np.nan_to_num(betas)


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

save_dir = os.path.join(args.project_dir, 'results', 'mds_nsdsynthetic_nsdcore')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'mds_zscore-' + str(args.zscore) + '_ncsnr_threshold-' + \
	str(args.ncsnr_threshold) + '.npy'

np.save(os.path.join(save_dir, file_name), results)
