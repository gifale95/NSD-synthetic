"""Prepare the NSD-core betas in fsaverage space. The betas are then divided
into train and test split for encoding models training and testing. The train
split consists of the fMRI reponses for the (up to) 9,000 subject unique
images. The test split consists of the fMRI responses for the 1,000 NSD-core
shared images that all subjects saw for three times.

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
from tqdm import tqdm
import pandas as pd
from scipy.stats import zscore
from nsdcode.nsd_mapdata import NSDmapdata # https://github.com/cvnlab/nsdcode
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Prepare NSD-core betas <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get order and ID of the presented images
# =============================================================================
# Load the experimental design info
nsd_expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsd', 'nsd_expdesign.mat'))
# Subtract 1 since the indices start with 1 (and not 0)
masterordering = nsd_expdesign['masterordering'] - 1
subjectim = nsd_expdesign['subjectim'] - 1

# Completed sessions per subject
if args.subject in (1, 2, 5, 7):
	sessions = 40
elif args.subject in (3, 6):
	sessions = 32
elif args.subject in (4, 8):
	sessions = 30

# Image presentation matrix of the selected subject
image_per_session = 750
tot_images = sessions * image_per_session
img_presentation_order = subjectim[args.subject-1,masterordering[0]][:tot_images]


# =============================================================================
# Load the ncsnr
# =============================================================================
lh_ncsnr = np.squeeze(nib.load(os.path.join(args.nsd_dir, 'nsddata_betas',
	'ppdata', 'subj'+format(args.subject, '02'), 'fsaverage',
	'betas_fithrf_GLMdenoise_RR', 'lh.ncsnr.mgh')).get_fdata())
lh_ncsnr = lh_ncsnr.astype(np.float32)
rh_ncsnr = np.squeeze(nib.load(os.path.join(args.nsd_dir, 'nsddata_betas',
	'ppdata', 'subj'+format(args.subject, '02'), 'fsaverage',
	'betas_fithrf_GLMdenoise_RR', 'rh.ncsnr.mgh')).get_fdata())
rh_ncsnr = rh_ncsnr.astype(np.float32)


# =============================================================================
# Prepare the fMRI betas
# =============================================================================
betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
	format(args.subject, '02'), 'fsaverage', 'betas_fithrf_GLMdenoise_RR')

for s in tqdm(range(sessions)):

	# Load the fMRI betas
	lh_file_name = 'lh.betas_session' + format(s+1, '02') + '.mgh'
	rh_file_name = 'rh.betas_session' + format(s+1, '02') + '.mgh'
	lh_betas_sess = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		lh_file_name)).get_fdata())).astype(np.float32)
	rh_betas_sess = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		rh_file_name)).get_fdata())).astype(np.float32)

	# Z-score the betas of each vertex within each scan session
	if args.zscore == 1:
		lh_betas_sess = zscore(lh_betas_sess, nan_policy='omit')
		rh_betas_sess = zscore(rh_betas_sess, nan_policy='omit')

	# Store the betas
	if s == 0:
		lh_betas = lh_betas_sess
		rh_betas = rh_betas_sess
	else:
		lh_betas = np.append(lh_betas, lh_betas_sess, 0)
		rh_betas = np.append(rh_betas, rh_betas_sess, 0)
	del lh_betas_sess, rh_betas_sess


# =============================================================================
# Save the train split betas
# =============================================================================
# Get all training image conditions presented in the available sessions
train_img_num = subjectim[args.subject-1,1000:]
train_img_num = train_img_num[np.isin(train_img_num, img_presentation_order)]
train_img_num.sort()

# Training betas array of shape: (Training conditions × Vertices)
lh_betas_train = np.zeros((len(train_img_num), lh_betas.shape[1]),
	dtype=np.float32)
rh_betas_train = np.zeros((len(train_img_num), rh_betas.shape[1]),
	dtype=np.float32)

# Average the train betas over repetitions
for i, img in enumerate(train_img_num):
	idx = np.where(img_presentation_order == img)[0]
	lh_betas_train[i] = np.nanmean(lh_betas[idx], 0)
	rh_betas_train[i] = np.nanmean(rh_betas[idx], 0)

# Save the train split betas
save_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
	'zscored-'+str(args.zscore), 'sub-0'+format(args.subject))
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
with h5py.File(os.path.join(save_dir, 'lh_betas_nsdcore_train.h5'), 'w') as f:
	f.create_dataset('betas', data=lh_betas_train, dtype=np.float32)
with h5py.File(os.path.join(save_dir, 'rh_betas_nsdcore_train.h5'), 'w') as f:
	f.create_dataset('betas', data=rh_betas_train, dtype=np.float32)
del lh_betas_train, rh_betas_train


# =============================================================================
# Save the test split betas
# =============================================================================
# Get all training image conditions presented in the available sessions
test_img_num = subjectim[args.subject-1,:1000]
test_img_num = test_img_num[np.isin(test_img_num, img_presentation_order)]
test_img_num.sort()

# Test betas array of shape: (Test conditions × Vertices)
lh_betas_test = np.zeros((len(test_img_num), lh_betas.shape[1]),
	dtype=np.float32)
rh_betas_test = np.zeros((len(test_img_num), rh_betas.shape[1]),
	dtype=np.float32)
test_img_repeats = np.zeros(len(test_img_num))

# Average the test betas over repetitions
for i, img in enumerate(test_img_num):
	idx = np.where(img_presentation_order == img)[0]
	lh_betas_test[i] = np.nanmean(lh_betas[idx], 0)
	rh_betas_test[i] = np.nanmean(rh_betas[idx], 0)
	test_img_repeats[i] = len(idx)

# Save the test split betas
with h5py.File(os.path.join(save_dir, 'lh_betas_nsdcore_test.h5'), 'w') as f:
	f.create_dataset('betas', data=lh_betas_test, dtype=np.float32)
with h5py.File(os.path.join(save_dir, 'rh_betas_nsdcore_test.h5'), 'w') as f:
	f.create_dataset('betas', data=rh_betas_test, dtype=np.float32)
del lh_betas_test, rh_betas_test

# Get the image conditions of the 515 NSD shared images with 3 repeats for each
# subject
min_sess = 30
min_images = min_sess * 750
min_img_presentation = img_presentation_order[:min_images]
test_part = subjectim[args.subject-1,:1000]
test_part = test_part[np.isin(test_part, min_img_presentation)]
test_part.sort()
test_img_num_special_515 = []
for i in range(len(test_part)):
	if len(np.where(min_img_presentation == test_part[i])[0]) == 3:
		test_img_num_special_515.append(test_part[i])
test_img_num_special_515 = np.asarray(test_img_num_special_515)


# =============================================================================
# Compute the ncsnr using 284 NSD shared image conditions
# =============================================================================
# Compute the ncsnr using the first 284 image conditions out of the 515/1000
# NSD shared images that all subjects saw for 3 times trials.

# Only select the first 284 image conditions with 3 repeats for each subject
# (to match the image condition number of NSD-synthetic)
conditions = test_img_num_special_515[:284]

# Estimate the noise standard deviation
# Calculate the variance of the betas across the three presentations of each
# image (using the unbiased estimator that normalizes by n–1 where n is the
# sample size)
lh_var = []
rh_var = []
for i, img in enumerate(conditions):
	idx = np.where(img_presentation_order == img)[0]
	if len(idx) != 3:
		raise Exception('Not 3 repeats!')
	lh_var.append(np.var(lh_betas[idx], axis=0, ddof=1))
	rh_var.append(np.var(rh_betas[idx], axis=0, ddof=1))
# Average the variance across images and compute the square root of the result
lh_sigma_noise = np.sqrt(np.mean(lh_var, 0))
rh_sigma_noise = np.sqrt(np.mean(rh_var, 0))

# Estimate the signal standard deviation
for i, img in enumerate(conditions):
	idx = np.where(img_presentation_order == img)[0]
	if i == 0:
		lh_betas_284 = lh_betas[idx]
		rh_betas_284 = rh_betas[idx]
	else:
		lh_betas_284 = np.append(lh_betas_284, lh_betas[idx], 0)
		rh_betas_284 = np.append(rh_betas_284, rh_betas[idx], 0)
if len(lh_betas_284) != 852:
	raise Exception('Not 852 trials!')
lh_var_data = np.var(lh_betas_284, axis=0, ddof=1)
rh_var_data = np.var(rh_betas_284, axis=0, ddof=1)
lh_sigma_signal = lh_var_data - (lh_sigma_noise ** 2)
rh_sigma_signal = rh_var_data - (rh_sigma_noise ** 2)
lh_sigma_signal[lh_sigma_signal<0] = 0
rh_sigma_signal[rh_sigma_signal<0] = 0
lh_sigma_signal = np.sqrt(lh_sigma_signal)
rh_sigma_signal = np.sqrt(rh_sigma_signal)

# Compute the ncsnr
lh_ncsnr_284 = lh_sigma_signal / lh_sigma_noise
rh_ncsnr_284 = rh_sigma_signal / rh_sigma_noise


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
	'train_img_num': train_img_num,
	'test_img_num': test_img_num,
	'test_img_repeats': test_img_repeats,
	'test_img_num_special_515': test_img_num_special_515,
	'lh_ncsnr': lh_ncsnr,
	'rh_ncsnr': rh_ncsnr,
	'lh_ncsnr_284': lh_ncsnr_284,
	'rh_ncsnr_284': rh_ncsnr_284,
	'lh_fsaverage_rois': lh_fsaverage_rois,
	'rh_fsaverage_rois': rh_fsaverage_rois
	}

np.save(os.path.join(save_dir, 'meatadata_nsdcore.npy'), metadata)
