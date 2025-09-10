"""Analyze the univariate and multivariate NSD-synthetic fMRI responses.

Parameters
----------
subjects : list
	List of the used NSD subjects.
rois : list
	List with all used NSD ROIs.
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
from tqdm import tqdm
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4',
	'PPA', 'VWFA'])
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> NSD-synthetic univariate/multivariate analyses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Subjects loop
# =============================================================================
retained_vertices = {}
univariate_responses = {}
rsms = {}
combinations = []

for roi in tqdm(args.rois):
	for sub in args.subjects:

		combinations.append('s'+str(sub)+'_'+roi)


# =============================================================================
# Load the metadata
# =============================================================================
		data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
			'zscore-'+str(args.zscore), 'sub-0'+format(sub))
		metadata = np.load(os.path.join(data_dir,
			'meatadata_nsdsynthetic.npy'), allow_pickle=True).item()
		lh_ncsnr = metadata['lh_ncsnr']
		rh_ncsnr = metadata['rh_ncsnr']


# =============================================================================
# Load the ROI indices
# =============================================================================
		# Merge ventral and dorsal V1/V2/V3
		if roi in ['V1', 'V2', 'V3']:
			lh_roi_idx = metadata['lh_fsaverage_rois'][roi+'v']
			lh_roi_idx = np.append(lh_roi_idx,
				metadata['lh_fsaverage_rois'][roi+'d'])
			lh_roi_idx.sort()
			rh_roi_idx = metadata['rh_fsaverage_rois'][roi+'v']
			rh_roi_idx = np.append(rh_roi_idx,
				metadata['rh_fsaverage_rois'][roi+'d'])
			rh_roi_idx.sort()
		# Merge anterior and posterior FBA/FFA/VWFA
		elif roi in ['FBA', 'FFA', 'VWFA']:
			lh_roi_idx = metadata['lh_fsaverage_rois'][roi+'-1']
			lh_roi_idx = np.append(lh_roi_idx,
				metadata['lh_fsaverage_rois'][roi+'-2'])
			lh_roi_idx.sort()
			rh_roi_idx = metadata['rh_fsaverage_rois'][roi+'-1']
			rh_roi_idx = np.append(rh_roi_idx,
				metadata['rh_fsaverage_rois'][roi+'-2'])
			rh_roi_idx.sort()
		# Other ROIs
		else:
			lh_roi_idx = metadata['lh_fsaverage_rois'][roi]
			rh_roi_idx = metadata['rh_fsaverage_rois'][roi]


# =============================================================================
# Only retain vertices above the ncsnr threshold
# =============================================================================
		lh_ncsnr_roi = lh_ncsnr[lh_roi_idx]
		rh_ncsnr_roi = rh_ncsnr[rh_roi_idx]

		lh_idx = lh_roi_idx[lh_ncsnr_roi>args.ncsnr_threshold]
		rh_idx = rh_roi_idx[rh_ncsnr_roi>args.ncsnr_threshold]


# =============================================================================
# Load the NSD-synthetic fMRI responses
# =============================================================================
		lh_betas = np.nan_to_num(h5py.File(os.path.join(data_dir,
			'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:,lh_idx])
		rh_betas = np.nan_to_num(h5py.File(os.path.join(data_dir,
			'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:,rh_idx])

		# Append the data from left and right hemispheres
		betas = np.append(lh_betas, rh_betas, 1)
		del lh_betas, rh_betas

		# Convert NaN values (missing data) to zeros
		betas = np.nan_to_num(betas)

		# Store the number of retained vertices
		retained_vertices['s'+str(sub)+'_'+roi] = betas.shape[1]


# =============================================================================
# Compute and store the univariate responses
# =============================================================================
		univariate_responses['s'+str(sub)+'_'+roi] = np.mean(betas, 1)


# =============================================================================
# Compute and store the RSMs
# =============================================================================
		rms = np.ones((len(betas), len(betas)), dtype=np.float32)

		for i1 in range(len(betas)):
			for i2 in range(i1):
				rms[i1,i2] = pearsonr(betas[i1], betas[i2])[0]
				rms[i2,i1] = rms[i1,i2]

		rsms['s'+str(sub)+'_'+roi] = rms

		del betas, rms


# =============================================================================
# Perform RSA
# =============================================================================
rsa = np.ones((len(combinations), len(combinations)), dtype=np.float32)

for i1, key_1 in enumerate(combinations):
	for i2, key_2 in enumerate(combinations):
		if i2 < i1:

			# Select the RSMs
			rms_1 = rsms[key_1]
			rms_2 = rsms[key_2]
			idx = np.tril_indices(len(rms_1), -1)
			rms_1 = rms_1[idx]
			rms_2 = rms_2[idx]

			# Correlate the RSMs
			rsa[i1,i2] = pearsonr(rms_1, rms_2)[0]
			rsa[i2,i1] = rsa[i1,i2]


# =============================================================================
# Save the results
# =============================================================================
results = {
	'retained_vertices': retained_vertices,
	'univariate_responses': univariate_responses,
	'rsms': rsms,
	'combinations': combinations,
	'rsa': rsa
	}

save_dir = os.path.join(args.project_dir, 'results', 'nsdsynthetic_responses',
	'zscore-'+str(args.zscore))
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'nsdsynthetic_responses_ncsnr_threshold-' + \
	format(args.ncsnr_threshold, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), results)
