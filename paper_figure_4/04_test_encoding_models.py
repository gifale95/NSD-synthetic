"""Test the encoding model's prediction accuracy in-distribution (using the
first 284 of 515/1,000 NSD-core's shared images that all subjects saw for 3
times), and out-of-distribution (using NSD-synthetic 284 images).

Parameters
----------
subjects_all : list
	List with all used NSD subject.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
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
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Test encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Empty result lists
# =============================================================================
lh_correlation_nsdcore_test = []
rh_correlation_nsdcore_test = []
lh_r2_nsdcore_test = []
rh_r2_nsdcore_test = []
lh_nc_nsdcore_test = []
rh_nc_nsdcore_test = []
lh_explained_variance_nsdcore_test = []
rh_explained_variance_nsdcore_test = []
lh_correlation_nsdsynthetic = []
rh_correlation_nsdsynthetic = []
lh_r2_nsdsynthetic = []
rh_r2_nsdsynthetic = []
lh_nc_nsdsynthetic = []
rh_nc_nsdsynthetic = []
lh_explained_variance_nsdsynthetic = []
rh_explained_variance_nsdsynthetic = []

for sub in tqdm(args.subjects):


# =============================================================================
# Load the recorded and predicted fMRI responses for the NSD-core test images
# =============================================================================
	# Load the metadata
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscored-'+str(args.zscore), 'sub-0'+format(sub),
		'meatadata_nsdcore.npy')
	metadata_nsdcore = np.load(data_dir, allow_pickle=True).item()

	# Only select the first 284 image conditions with 3 repeats for each
	# subjects (to match the image condition number of NSD-synthetic)
	idx = np.where(np.isin(metadata_nsdcore['test_img_num'],
		metadata_nsdcore['test_img_num_special_515']))[0][:284]

	# Recorded fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscored-'+str(args.zscore), 'sub-0'+format(sub))
	lh_betas_nsdcore_test = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdcore_test.h5'), 'r')['betas'][idx,:]
	rh_betas_nsdcore_test = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdcore_test.h5'), 'r')['betas'][idx,:]
	# Set NaN values (missing fMRI data) to zero
	lh_betas_nsdcore_test = np.nan_to_num(lh_betas_nsdcore_test)
	rh_betas_nsdcore_test = np.nan_to_num(rh_betas_nsdcore_test)

	# Predicted fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'predicted_fmri',
		'zscored-'+str(args.zscore), 'model-vit_b_32',
		'predicted_fmri_sub-0'+str(sub)+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	lh_betas_nsdcore_test_pred = data['lh_betas_nsdcore_test_pred'][idx,:]
	rh_betas_nsdcore_test_pred = data['rh_betas_nsdcore_test_pred'][idx,:]

	# ncsnr (284 image conditions)
	lh_ncsnr_nsdcore_test = metadata_nsdcore['lh_ncsnr_284']
	rh_ncsnr_nsdcore_test = metadata_nsdcore['rh_ncsnr_284']

	# Convert the ncsnr to noise ceiling (284 image conditions)
	norm_term = 1 / 3
	lh_nc_nsdcore_test_sub = (lh_ncsnr_nsdcore_test ** 2) / \
		((lh_ncsnr_nsdcore_test ** 2) + norm_term)
	rh_nc_nsdcore_test_sub = (rh_ncsnr_nsdcore_test ** 2) / \
		((rh_ncsnr_nsdcore_test ** 2) + norm_term)
	lh_nc_nsdcore_test.append(lh_nc_nsdcore_test_sub)
	rh_nc_nsdcore_test.append(rh_nc_nsdcore_test_sub)


# =============================================================================
# Load the recorded and predicted fMRI responses for the NSD-synthetic images
# =============================================================================
	# Recorded fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscored-'+str(args.zscore), 'sub-0'+format(sub))
	lh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:]
	rh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:]
	# Set NaN values (missing fMRI data) to zero
	lh_betas_nsdsynthetic = np.nan_to_num(lh_betas_nsdsynthetic)
	rh_betas_nsdsynthetic = np.nan_to_num(rh_betas_nsdsynthetic)

	# Predicted fMRI
	lh_betas_nsdsynthetic_pred = data['lh_betas_nsdsynthetic_pred']
	rh_betas_nsdsynthetic_pred = data['rh_betas_nsdsynthetic_pred']
	del data

	# ncsnr
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscored-'+str(args.zscore), 'sub-0'+format(sub),
		'meatadata_nsdsynthetic.npy')
	metadata_nsdsynthetic = np.load(data_dir, allow_pickle=True).item()
	lh_ncsnr_nsdsynthetic = metadata_nsdsynthetic['lh_ncsnr']
	rh_ncsnr_nsdsynthetic = metadata_nsdsynthetic['rh_ncsnr']

	# Convert the ncsnr to noise ceiling
	img_reps_2 = 236
	img_reps_4 = 32
	img_reps_8 = 8
	img_reps_10 = 8
	norm_term = (img_reps_2/2 + img_reps_4/4 + img_reps_8/8 + img_reps_10/10) / \
		(img_reps_2 + img_reps_4 + img_reps_8 + img_reps_10)
	lh_nc_nsdsynthetic_sub = (lh_ncsnr_nsdsynthetic ** 2) / \
		((lh_ncsnr_nsdsynthetic ** 2) + norm_term)
	rh_nc_nsdsynthetic_sub = (rh_ncsnr_nsdsynthetic ** 2) / \
		((rh_ncsnr_nsdsynthetic ** 2) + norm_term)
	lh_nc_nsdsynthetic.append(lh_nc_nsdsynthetic_sub)
	rh_nc_nsdsynthetic.append(rh_nc_nsdsynthetic_sub)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
	# Correlate the recorded and predicted fMRI responses
	lh_correlation_nsdcore_test_sub = np.zeros(
		lh_betas_nsdcore_test_pred.shape[1])
	rh_correlation_nsdcore_test_sub = np.zeros(
		rh_betas_nsdcore_test_pred.shape[1])
	lh_correlation_nsdsynthetic_sub = np.zeros(
		lh_betas_nsdsynthetic_pred.shape[1])
	rh_correlation_nsdsynthetic_sub = np.zeros(
		rh_betas_nsdsynthetic_pred.shape[1])
	for v in range(lh_betas_nsdcore_test_pred.shape[1]):
		lh_correlation_nsdcore_test_sub[v] = pearsonr(
			lh_betas_nsdcore_test[:,v], lh_betas_nsdcore_test_pred[:,v])[0]
		rh_correlation_nsdcore_test_sub[v] = pearsonr(
			rh_betas_nsdcore_test[:,v], rh_betas_nsdcore_test_pred[:,v])[0]
		lh_correlation_nsdsynthetic_sub[v] = pearsonr(
			lh_betas_nsdsynthetic[:,v], lh_betas_nsdsynthetic_pred[:,v])[0]
		rh_correlation_nsdsynthetic_sub[v] = pearsonr(
			rh_betas_nsdsynthetic[:,v], rh_betas_nsdsynthetic_pred[:,v])[0]
	# Set negative correlation scores to zero
	lh_correlation_nsdcore_test_sub[lh_correlation_nsdcore_test_sub<0] = 0
	rh_correlation_nsdcore_test_sub[rh_correlation_nsdcore_test_sub<0] = 0
	lh_correlation_nsdsynthetic_sub[lh_correlation_nsdsynthetic_sub<0] = 0
	rh_correlation_nsdsynthetic_sub[rh_correlation_nsdsynthetic_sub<0] = 0
	# Store the results
	lh_correlation_nsdcore_test.append(lh_correlation_nsdcore_test_sub)
	rh_correlation_nsdcore_test.append(rh_correlation_nsdcore_test_sub)
	lh_correlation_nsdsynthetic.append(lh_correlation_nsdsynthetic_sub)
	rh_correlation_nsdsynthetic.append(rh_correlation_nsdsynthetic_sub)
	del lh_betas_nsdcore_test, rh_betas_nsdcore_test, \
		lh_betas_nsdcore_test_pred, rh_betas_nsdcore_test_pred, \
		lh_betas_nsdsynthetic, rh_betas_nsdsynthetic, \
		lh_betas_nsdsynthetic_pred, rh_betas_nsdsynthetic_pred

	# Turn the correlations into r2 scores
	lh_r2_nsdcore_test_sub = lh_correlation_nsdcore_test_sub ** 2
	rh_r2_nsdcore_test_sub = rh_correlation_nsdcore_test_sub ** 2
	lh_r2_nsdsynthetic_sub = lh_correlation_nsdsynthetic_sub ** 2
	rh_r2_nsdsynthetic_sub = rh_correlation_nsdsynthetic_sub ** 2
	lh_r2_nsdcore_test.append(lh_r2_nsdcore_test_sub)
	rh_r2_nsdcore_test.append(rh_r2_nsdcore_test_sub)
	lh_r2_nsdsynthetic.append(lh_r2_nsdsynthetic_sub)
	rh_r2_nsdsynthetic.append(rh_r2_nsdsynthetic_sub)

	# Add a very small number to noise ceiling values of 0, otherwise the
	# noise-ceiling-normalized encoding accuracy cannot be calculated (division
	# by 0 is not possible)
	lh_nc_nsdcore_test_sub[lh_nc_nsdcore_test_sub==0] = 1e-14
	rh_nc_nsdcore_test_sub[rh_nc_nsdcore_test_sub==0] = 1e-14
	lh_nc_nsdsynthetic_sub[lh_nc_nsdsynthetic_sub==0] = 1e-14
	rh_nc_nsdsynthetic_sub[rh_nc_nsdsynthetic_sub==0] = 1e-14

	# Compute the noise-ceiling-normalized encoding accuracy
	lh_explained_variance_nsdcore_test_sub = np.divide(lh_r2_nsdcore_test_sub,
		lh_nc_nsdcore_test_sub) * 100
	rh_explained_variance_nsdcore_test_sub = np.divide(rh_r2_nsdcore_test_sub,
		rh_nc_nsdcore_test_sub) * 100
	lh_explained_variance_nsdsynthetic_sub = np.divide(lh_r2_nsdsynthetic_sub,
		lh_nc_nsdsynthetic_sub) * 100
	rh_explained_variance_nsdsynthetic_sub = np.divide(rh_r2_nsdsynthetic_sub,
		rh_nc_nsdsynthetic_sub) * 100

	# Set the noise-normalized encoding accuracy to 100 for vertices where the
	# the correlation is higher than the noise ceiling, to prevent encoding
	# accuracy values higher than 100%
	lh_explained_variance_nsdcore_test_sub\
		[lh_explained_variance_nsdcore_test_sub>100] = 100
	rh_explained_variance_nsdcore_test_sub\
		[rh_explained_variance_nsdcore_test_sub>100] = 100
	lh_explained_variance_nsdsynthetic_sub\
		[lh_explained_variance_nsdsynthetic_sub>100] = 100
	rh_explained_variance_nsdsynthetic_sub\
		[rh_explained_variance_nsdsynthetic_sub>100] = 100

	# Store the encoding accuracy results
	lh_explained_variance_nsdcore_test.append(
		lh_explained_variance_nsdcore_test_sub)
	rh_explained_variance_nsdcore_test.append(
		rh_explained_variance_nsdcore_test_sub)
	lh_explained_variance_nsdsynthetic.append(
		lh_explained_variance_nsdsynthetic_sub)
	rh_explained_variance_nsdsynthetic.append(
		rh_explained_variance_nsdsynthetic_sub)

	# Delete unused variables
	del lh_correlation_nsdcore_test_sub, rh_correlation_nsdcore_test_sub, \
		lh_r2_nsdcore_test_sub, rh_r2_nsdcore_test_sub, \
		lh_nc_nsdcore_test_sub, rh_nc_nsdcore_test_sub, \
		lh_explained_variance_nsdcore_test_sub, \
		rh_explained_variance_nsdcore_test_sub, \
		lh_correlation_nsdsynthetic_sub, rh_correlation_nsdsynthetic_sub, \
		lh_r2_nsdsynthetic_sub, rh_r2_nsdsynthetic_sub, \
		lh_nc_nsdsynthetic_sub, rh_nc_nsdsynthetic_sub, \
		lh_explained_variance_nsdsynthetic_sub, \
		rh_explained_variance_nsdsynthetic_sub


# =============================================================================
# Save the prediction accuracy
# =============================================================================
results = {
	'lh_correlation_nsdcore_test': lh_correlation_nsdcore_test,
	'rh_correlation_nsdcore_test': rh_correlation_nsdcore_test,
	'lh_r2_nsdcore_test': lh_r2_nsdcore_test,
	'rh_r2_nsdcore_test': rh_r2_nsdcore_test,
	'lh_nc_nsdcore_test': lh_nc_nsdcore_test,
	'rh_nc_nsdcore_test': rh_nc_nsdcore_test,
	'lh_explained_variance_nsdcore_test' : lh_explained_variance_nsdcore_test,
	'rh_explained_variance_nsdcore_test': rh_explained_variance_nsdcore_test,
	'lh_correlation_nsdsynthetic' : lh_correlation_nsdsynthetic,
	'rh_correlation_nsdsynthetic': rh_correlation_nsdsynthetic,
	'lh_r2_nsdsynthetic' : lh_r2_nsdsynthetic,
	'rh_r2_nsdsynthetic': rh_r2_nsdsynthetic,
	'lh_nc_nsdsynthetic' : lh_nc_nsdsynthetic,
	'rh_nc_nsdsynthetic': rh_nc_nsdsynthetic,
	'lh_explained_variance_nsdsynthetic' : lh_explained_variance_nsdsynthetic,
	'rh_explained_variance_nsdsynthetic': rh_explained_variance_nsdsynthetic
	}

save_dir = os.path.join(args.project_dir, 'results', 'encoding_accuracy',
	'zscored-'+str(args.zscore), 'model-vit_b_32')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)
