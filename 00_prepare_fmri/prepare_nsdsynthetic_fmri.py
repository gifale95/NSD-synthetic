"""Prepare the NSD-synthetic betas in fsaverage space.

Parameters
----------
subject : int
	Number of the used NSD subject.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
nsd_dir : str
	Directory of the NSD.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from scipy.io import loadmat
import nibabel as nib
from scipy.stats import zscore
import pandas as pd
from nsdcode.nsd_mapdata import NSDmapdata # https://github.com/cvnlab/nsdcode
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Prepare NSD-synthetic betas <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get order and ID of the presented images
# =============================================================================
# Load the experimental design info
expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsynthetic_expdesign.mat'))
# Subtract 1 since the indices start with 1 (and not 0)
masterordering = np.squeeze(expdesign['masterordering'] - 1)


# =============================================================================
# Prepare the fMRI betas
# =============================================================================
betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
	format(args.subject, '02'), 'fsaverage',
	'nsdsyntheticbetas_fithrf_GLMdenoise_RR')

# Load the fMRI betas
lh_file_name = 'lh.betas_nsdsynthetic.mgh'
rh_file_name = 'rh.betas_nsdsynthetic.mgh'
lh_betas_all = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
	lh_file_name)).get_fdata())).astype(np.float32)
rh_betas_all = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
	rh_file_name)).get_fdata())).astype(np.float32)

# z-score the betas of each vertex within the scan session
if args.zscore == 1:
	lh_betas_all = zscore(lh_betas_all, nan_policy='omit')
	rh_betas_all = zscore(rh_betas_all, nan_policy='omit')


# =============================================================================
# Get the NSD synthetic image condition repeats
# =============================================================================
nsdsynthetic_img_num = np.unique(masterordering)
nsdsynthetic_img_repeats = np.zeros(len(nsdsynthetic_img_num))

for i, img in enumerate(nsdsynthetic_img_num):
	idx = np.where(masterordering == img)[0]
	nsdsynthetic_img_repeats[i] = len(idx)


# =============================================================================
# Compute the ncsnr
# =============================================================================
# When computing the ncsnr on image conditions with different amounts of trials
# (i.e., different sample sizes), I need to correct for this:
# https://stats.stackexchange.com/questions/488911/combined-variance-estimate-for-samples-of-varying-sizes

lh_num_var = np.zeros((lh_betas_all.shape[1]))
rh_num_var = np.zeros((rh_betas_all.shape[1]))
den_var = np.zeros((lh_betas_all.shape[1]))

for i, img in enumerate(nsdsynthetic_img_num):
	idx = np.where(masterordering == img)[0]
	lh_num_var += np.var(lh_betas_all[idx], axis=0, ddof=1) * (len(idx) - 1)
	rh_num_var += np.var(rh_betas_all[idx], axis=0, ddof=1) * (len(idx) - 1)
	den_var += len(idx) - 1

lh_sigma_noise = np.sqrt(lh_num_var/den_var)
rh_sigma_noise = np.sqrt(rh_num_var/den_var)
lh_var_data = np.var(lh_betas_all, axis=0, ddof=1)
rh_var_data = np.var(rh_betas_all, axis=0, ddof=1)
lh_sigma_signal = lh_var_data - (lh_sigma_noise ** 2)
rh_sigma_signal = rh_var_data - (rh_sigma_noise ** 2)
lh_sigma_signal[lh_sigma_signal<0] = 0
rh_sigma_signal[rh_sigma_signal<0] = 0
lh_sigma_signal = np.sqrt(lh_sigma_signal)
rh_sigma_signal = np.sqrt(rh_sigma_signal)
lh_ncsnr = lh_sigma_signal / lh_sigma_noise
rh_ncsnr = rh_sigma_signal / rh_sigma_noise


# =============================================================================
# Average the fMRI across repeats, and save
# =============================================================================
# Average across repeats
lh_betas = np.zeros((len(nsdsynthetic_img_num), lh_betas_all.shape[1]))
rh_betas = np.zeros((len(nsdsynthetic_img_num), rh_betas_all.shape[1]))
for i, img in enumerate(nsdsynthetic_img_num):
	idx = np.where(masterordering == img)[0]
	lh_betas[i] = np.nanmean(lh_betas_all[idx], 0)
	rh_betas[i] = np.nanmean(rh_betas_all[idx], 0)

# Save the betas
save_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
	'zscored-'+str(args.zscore), 'sub-0'+format(args.subject))
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
with h5py.File(os.path.join(save_dir, 'lh_betas_nsdsynthetic.h5'), 'w') as f:
	f.create_dataset('betas', data=lh_betas, dtype=np.float32)
with h5py.File(os.path.join(save_dir, 'rh_betas_nsdsynthetic.h5'), 'w') as f:
	f.create_dataset('betas', data=rh_betas, dtype=np.float32)


# =============================================================================
# Prepare the ROI mask indices
# =============================================================================
# Save the mapping between ROI names and ROI mask values
roi_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
	format(args.subject, '02'), 'label')
roi_map_files = ['prf-visualrois.mgz.ctab', 'floc-bodies.mgz.ctab',
	'floc-faces.mgz.ctab', 'floc-places.mgz.ctab', 'floc-words.mgz.ctab',
	'streams.mgz.ctab']
roi_name_maps = []
for r in roi_map_files:
	roi_map = pd.read_csv(os.path.join(roi_dir, r), delimiter=' ',
		header=None, index_col=0)
	roi_map = roi_map.to_dict()[1]
	roi_name_maps.append(roi_map)

# Map the ROI mask indices from subject native space to fsaverage space
lh_roi_files = ['lh.prf-visualrois.mgz', 'lh.floc-bodies.mgz',
	'lh.floc-faces.mgz', 'lh.floc-places.mgz', 'lh.floc-words.mgz',
	'lh.streams.mgz']
rh_roi_files = ['rh.prf-visualrois.mgz', 'rh.floc-bodies.mgz',
	'rh.floc-faces.mgz', 'rh.floc-places.mgz', 'rh.floc-words.mgz',
	'rh.streams.mgz']
# Initiate NSDmapdata
nsd = NSDmapdata(args.nsd_dir)
lh_fsaverage_rois = {}
rh_fsaverage_rois = {}
for r1 in range(len(lh_roi_files)):
	# Map the ROI masks from subject native to fsaverage space
	lh_fsaverage_roi = np.squeeze(nsd.fit(args.subject, 'lh.white',
		'fsaverage', os.path.join(roi_dir, lh_roi_files[r1])))
	rh_fsaverage_roi = np.squeeze(nsd.fit(args.subject, 'rh.white',
		'fsaverage', os.path.join(roi_dir, rh_roi_files[r1])))
	# Store the ROI masks
	for r2 in roi_name_maps[r1].items():
		if r2[0] != 0:
			lh_fsaverage_rois[r2[1]] = np.where(lh_fsaverage_roi == r2[0])[0]
			rh_fsaverage_rois[r2[1]] = np.where(rh_fsaverage_roi == r2[0])[0]


# =============================================================================
# Save the metadata
# =============================================================================
metadata = {
	'masterordering': masterordering,
	'nsdsynthetic_img_num': nsdsynthetic_img_num,
	'nsdsynthetic_img_repeats': nsdsynthetic_img_repeats,
	'lh_ncsnr': lh_ncsnr,
	'rh_ncsnr': rh_ncsnr,
	'lh_fsaverage_rois': lh_fsaverage_rois,
	'rh_fsaverage_rois': rh_fsaverage_rois,
	}

np.save(os.path.join(save_dir, 'meatadata_nsdsynthetic.npy'), metadata)
