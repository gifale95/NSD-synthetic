"""Prepare the NSD-core betas in fsaverage space.

Parameters
----------
subject : int
	Number of the used NSD subject.
train_test_session_control : int
	If '1', use the train and test splits consist of image conditions from
	non-overlapping fMRI scan sessions.
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
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
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
img_num = np.unique(img_presentation_order)
img_num.sort()


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
# Save the betas
# =============================================================================
# Betas array of shape: (Image conditions × Vertices)
lh_betas_avg = np.zeros((len(img_num), lh_betas.shape[1]), dtype=np.float32)
rh_betas_avg = np.zeros((len(img_num), rh_betas.shape[1]), dtype=np.float32)
img_repeats = np.zeros(len(img_num))

# Average the betas over repetitions
for i, img in enumerate(img_num):
	idx = np.where(img_presentation_order == img)[0]
	lh_betas_avg[i] = np.nanmean(lh_betas[idx], 0)
	rh_betas_avg[i] = np.nanmean(rh_betas[idx], 0)
	img_repeats[i] = len(idx)

# Set NaN values (missing fMRI data) to zero
lh_betas_avg = np.nan_to_num(lh_betas_avg)
rh_betas_avg = np.nan_to_num(rh_betas_avg)

# Save the betas
save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+format(args.subject))
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
with h5py.File(os.path.join(save_dir, 'lh_betas_nsdcore.h5'), 'w') as f:
	f.create_dataset('betas', data=lh_betas_avg, dtype=np.float32)
with h5py.File(os.path.join(save_dir, 'rh_betas_nsdcore.h5'), 'w') as f:
	f.create_dataset('betas', data=rh_betas_avg, dtype=np.float32)
del lh_betas_avg, rh_betas_avg


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
	'img_num': img_num,
	'img_repeats': img_repeats,
	'lh_ncsnr': lh_ncsnr,
	'rh_ncsnr': rh_ncsnr,
	'lh_fsaverage_rois': lh_fsaverage_rois,
	'rh_fsaverage_rois': rh_fsaverage_rois
	}

np.save(os.path.join(save_dir, 'metadata_nsdcore.npy'), metadata)
