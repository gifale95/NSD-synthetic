"""Test the encoding model's prediction accuracy ID and OOD on NSD-core, and
OOD on NSD-synthetic.

Parameters
----------
subjects : list
	List of the used NSD subjects.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet', 'resnet50', 'moco', and 'vit_b_32'.
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
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Test encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Empty result lists
# =============================================================================
lh_correlation_nsdcore_test_id = []
rh_correlation_nsdcore_test_id = []
lh_r2_nsdcore_test_id = []
rh_r2_nsdcore_test_id = []
lh_nc_nsdcore_test_id = []
rh_nc_nsdcore_test_id = []
lh_explained_variance_nsdcore_test_id = []
rh_explained_variance_nsdcore_test_id = []
lh_correlation_nsdcore_test_ood = []
rh_correlation_nsdcore_test_ood = []
lh_r2_nsdcore_test_ood = []
rh_r2_nsdcore_test_ood = []
lh_nc_nsdcore_test_ood = []
rh_nc_nsdcore_test_ood = []
lh_explained_variance_nsdcore_test_ood = []
rh_explained_variance_nsdcore_test_ood = []
lh_correlation_nsdsynthetic = []
rh_correlation_nsdsynthetic = []
lh_r2_nsdsynthetic = []
rh_r2_nsdsynthetic = []
lh_nc_nsdsynthetic = []
rh_nc_nsdsynthetic = []
lh_explained_variance_nsdsynthetic = []
rh_explained_variance_nsdsynthetic = []

# Loop across subjects
for sub in tqdm(args.subjects):


# =============================================================================
# Load the recorded and predicted fMRI responses for the NSD-core test images
# =============================================================================
	# Load the metadata
	data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+str(sub),
		'meatadata_nsdcore.npy')
	metadata_nsdcore = np.load(data_dir, allow_pickle=True).item()

	# Recorded fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+str(sub))
	lh_betas_nsdcore_test_id = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdcore_test_id.h5'), 'r')['betas'][:]
	rh_betas_nsdcore_test_id = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdcore_test_id.h5'), 'r')['betas'][:]
	lh_betas_nsdcore_test_ood = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdcore_test_ood.h5'), 'r')['betas'][:]
	rh_betas_nsdcore_test_ood = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdcore_test_ood.h5'), 'r')['betas'][:]

	# Predicted fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'predicted_fmri', 'zscore-'+str(args.zscore), 'model-'+args.model,
		'predicted_fmri_sub-0'+str(sub)+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	lh_betas_nsdcore_test_id_pred = data['lh_betas_nsdcore_test_id_pred']
	rh_betas_nsdcore_test_id_pred = data['rh_betas_nsdcore_test_id_pred']
	lh_betas_nsdcore_test_ood_pred = data['lh_betas_nsdcore_test_ood_pred']
	rh_betas_nsdcore_test_ood_pred = data['rh_betas_nsdcore_test_ood_pred']

	# Convert the ncsnr to noise ceiling (NSD-core ID test images)
	lh_ncsnr_nsdcore_test_id = metadata_nsdcore['lh_ncsnr_id']
	rh_ncsnr_nsdcore_test_id = metadata_nsdcore['rh_ncsnr_id']
	test_img_repeats = \
		metadata_nsdcore['train_test_splits']['test_img_id_repeats']
	img_reps_1 = len(np.where(test_img_repeats == 1)[0])
	img_reps_2 = len(np.where(test_img_repeats == 2)[0])
	img_reps_3 = len(np.where(test_img_repeats == 3)[0])
	norm_term = (img_reps_1/1 + img_reps_2/2 + img_reps_3/3) / \
		(img_reps_1 + img_reps_2 + img_reps_3)
	lh_nc_nsdcore_test_id_sub = (lh_ncsnr_nsdcore_test_id ** 2) / \
		((lh_ncsnr_nsdcore_test_id ** 2) + norm_term).astype(np.float32)
	rh_nc_nsdcore_test_id_sub = (rh_ncsnr_nsdcore_test_id ** 2) / \
		((rh_ncsnr_nsdcore_test_id ** 2) + norm_term).astype(np.float32)
	lh_nc_nsdcore_test_id.append(lh_nc_nsdcore_test_id_sub)
	rh_nc_nsdcore_test_id.append(rh_nc_nsdcore_test_id_sub)

	# Convert the ncsnr to noise ceiling (NSD-core OOD test images)
	lh_ncsnr_nsdcore_test_ood = metadata_nsdcore['lh_ncsnr_ood']
	rh_ncsnr_nsdcore_test_ood = metadata_nsdcore['rh_ncsnr_ood']
	test_img_repeats = \
		metadata_nsdcore['train_test_splits']['test_img_ood_repeats']
	img_reps_1 = len(np.where(test_img_repeats == 1)[0])
	img_reps_2 = len(np.where(test_img_repeats == 2)[0])
	img_reps_3 = len(np.where(test_img_repeats == 3)[0])
	norm_term = (img_reps_1/1 + img_reps_2/2 + img_reps_3/3) / \
		(img_reps_1 + img_reps_2 + img_reps_3)
	lh_nc_nsdcore_test_ood_sub = (lh_ncsnr_nsdcore_test_ood ** 2) / \
		((lh_ncsnr_nsdcore_test_ood ** 2) + norm_term).astype(np.float32)
	rh_nc_nsdcore_test_ood_sub = (rh_ncsnr_nsdcore_test_ood ** 2) / \
		((rh_ncsnr_nsdcore_test_ood ** 2) + norm_term).astype(np.float32)
	lh_nc_nsdcore_test_ood.append(lh_nc_nsdcore_test_ood_sub)
	rh_nc_nsdcore_test_ood.append(rh_nc_nsdcore_test_ood_sub)


# =============================================================================
# Load the recorded and predicted fMRI responses for the NSD-synthetic images
# =============================================================================
	# Recorded fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+str(sub))
	lh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:]
	rh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:]

	# Predicted fMRI
	lh_betas_nsdsynthetic_pred = data['lh_betas_nsdsynthetic_pred']
	rh_betas_nsdsynthetic_pred = data['rh_betas_nsdsynthetic_pred']
	del data

	# ncsnr
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+str(sub),
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
	lh_nc_nsdsynthetic_sub = ((lh_ncsnr_nsdsynthetic ** 2) / \
		((lh_ncsnr_nsdsynthetic ** 2) + norm_term)).astype(np.float32)
	rh_nc_nsdsynthetic_sub = ((rh_ncsnr_nsdsynthetic ** 2) / \
		((rh_ncsnr_nsdsynthetic ** 2) + norm_term)).astype(np.float32)
	lh_nc_nsdsynthetic.append(lh_nc_nsdsynthetic_sub)
	rh_nc_nsdsynthetic.append(rh_nc_nsdsynthetic_sub)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
	# Correlate the recorded and predicted fMRI responses
	lh_correlation_nsdcore_test_id_sub = np.zeros(
		lh_betas_nsdcore_test_id_pred.shape[1], dtype=np.float32)
	rh_correlation_nsdcore_test_id_sub = np.zeros(
		rh_betas_nsdcore_test_id_pred.shape[1], dtype=np.float32)
	lh_correlation_nsdcore_test_ood_sub = np.zeros(
		lh_betas_nsdcore_test_ood_pred.shape[1], dtype=np.float32)
	rh_correlation_nsdcore_test_ood_sub = np.zeros(
		rh_betas_nsdcore_test_ood_pred.shape[1], dtype=np.float32)
	lh_correlation_nsdsynthetic_sub = np.zeros(
		lh_betas_nsdsynthetic_pred.shape[1], dtype=np.float32)
	rh_correlation_nsdsynthetic_sub = np.zeros(
		rh_betas_nsdsynthetic_pred.shape[1], dtype=np.float32)
	for v in range(lh_betas_nsdcore_test_id_pred.shape[1]):
		lh_correlation_nsdcore_test_id_sub[v] = pearsonr(
			lh_betas_nsdcore_test_id[:,v],
			lh_betas_nsdcore_test_id_pred[:,v])[0]
		rh_correlation_nsdcore_test_id_sub[v] = pearsonr(
			rh_betas_nsdcore_test_id[:,v],
			rh_betas_nsdcore_test_id_pred[:,v])[0]
		lh_correlation_nsdcore_test_ood_sub[v] = pearsonr(
			lh_betas_nsdcore_test_ood[:,v],
			lh_betas_nsdcore_test_ood_pred[:,v])[0]
		rh_correlation_nsdcore_test_ood_sub[v] = pearsonr(
			rh_betas_nsdcore_test_ood[:,v],
			rh_betas_nsdcore_test_ood_pred[:,v])[0]
		lh_correlation_nsdsynthetic_sub[v] = pearsonr(
			lh_betas_nsdsynthetic[:,v], lh_betas_nsdsynthetic_pred[:,v])[0]
		rh_correlation_nsdsynthetic_sub[v] = pearsonr(
			rh_betas_nsdsynthetic[:,v], rh_betas_nsdsynthetic_pred[:,v])[0]
	# Set negative correlation scores to zero
	lh_correlation_nsdcore_test_id_sub[lh_correlation_nsdcore_test_id_sub<0] = 0
	rh_correlation_nsdcore_test_id_sub[rh_correlation_nsdcore_test_id_sub<0] = 0
	lh_correlation_nsdcore_test_ood_sub[lh_correlation_nsdcore_test_ood_sub<0] = 0
	rh_correlation_nsdcore_test_ood_sub[rh_correlation_nsdcore_test_ood_sub<0] = 0
	lh_correlation_nsdsynthetic_sub[lh_correlation_nsdsynthetic_sub<0] = 0
	rh_correlation_nsdsynthetic_sub[rh_correlation_nsdsynthetic_sub<0] = 0
	# Store the results
	lh_correlation_nsdcore_test_id.append(lh_correlation_nsdcore_test_id_sub)
	rh_correlation_nsdcore_test_id.append(rh_correlation_nsdcore_test_id_sub)
	lh_correlation_nsdcore_test_ood.append(lh_correlation_nsdcore_test_ood_sub)
	rh_correlation_nsdcore_test_ood.append(rh_correlation_nsdcore_test_ood_sub)
	lh_correlation_nsdsynthetic.append(lh_correlation_nsdsynthetic_sub)
	rh_correlation_nsdsynthetic.append(rh_correlation_nsdsynthetic_sub)
	del lh_betas_nsdcore_test_id, rh_betas_nsdcore_test_id, \
		lh_betas_nsdcore_test_ood, rh_betas_nsdcore_test_ood, \
		lh_betas_nsdcore_test_id_pred, rh_betas_nsdcore_test_id_pred, \
			lh_betas_nsdcore_test_ood_pred, rh_betas_nsdcore_test_ood_pred, \
		lh_betas_nsdsynthetic, rh_betas_nsdsynthetic, \
		lh_betas_nsdsynthetic_pred, rh_betas_nsdsynthetic_pred

	# Turn the correlations into r2 scores
	lh_r2_nsdcore_test_id_sub = lh_correlation_nsdcore_test_id_sub ** 2
	rh_r2_nsdcore_test_id_sub = rh_correlation_nsdcore_test_id_sub ** 2
	lh_r2_nsdcore_test_ood_sub = lh_correlation_nsdcore_test_ood_sub ** 2
	rh_r2_nsdcore_test_ood_sub = rh_correlation_nsdcore_test_ood_sub ** 2
	lh_r2_nsdsynthetic_sub = lh_correlation_nsdsynthetic_sub ** 2
	rh_r2_nsdsynthetic_sub = rh_correlation_nsdsynthetic_sub ** 2
	lh_r2_nsdcore_test_id.append(lh_r2_nsdcore_test_id_sub)
	rh_r2_nsdcore_test_id.append(rh_r2_nsdcore_test_id_sub)
	lh_r2_nsdcore_test_ood.append(lh_r2_nsdcore_test_ood_sub)
	rh_r2_nsdcore_test_ood.append(rh_r2_nsdcore_test_ood_sub)
	lh_r2_nsdsynthetic.append(lh_r2_nsdsynthetic_sub)
	rh_r2_nsdsynthetic.append(rh_r2_nsdsynthetic_sub)

	# Add a very small number to noise ceiling values of 0, otherwise the
	# noise-ceiling-normalized encoding accuracy cannot be calculated (division
	# by 0 is not possible)
	lh_nc_nsdcore_test_id_sub[lh_nc_nsdcore_test_id_sub==0] = 1e-14
	rh_nc_nsdcore_test_id_sub[rh_nc_nsdcore_test_id_sub==0] = 1e-14
	lh_nc_nsdcore_test_ood_sub[lh_nc_nsdcore_test_ood_sub==0] = 1e-14
	rh_nc_nsdcore_test_ood_sub[rh_nc_nsdcore_test_ood_sub==0] = 1e-14
	lh_nc_nsdsynthetic_sub[lh_nc_nsdsynthetic_sub==0] = 1e-14
	rh_nc_nsdsynthetic_sub[rh_nc_nsdsynthetic_sub==0] = 1e-14

	# Compute the noise-ceiling-normalized encoding accuracy
	lh_explained_variance_nsdcore_test_id_sub = np.divide(
		lh_r2_nsdcore_test_id_sub, lh_nc_nsdcore_test_id_sub) * 100
	rh_explained_variance_nsdcore_test_id_sub = np.divide(
		rh_r2_nsdcore_test_id_sub, rh_nc_nsdcore_test_id_sub) * 100
	lh_explained_variance_nsdcore_test_ood_sub = np.divide(
		lh_r2_nsdcore_test_ood_sub, lh_nc_nsdcore_test_ood_sub) * 100
	rh_explained_variance_nsdcore_test_ood_sub = np.divide(
		rh_r2_nsdcore_test_ood_sub, rh_nc_nsdcore_test_ood_sub) * 100
	lh_explained_variance_nsdsynthetic_sub = np.divide(lh_r2_nsdsynthetic_sub,
		lh_nc_nsdsynthetic_sub) * 100
	rh_explained_variance_nsdsynthetic_sub = np.divide(rh_r2_nsdsynthetic_sub,
		rh_nc_nsdsynthetic_sub) * 100

	# Set the noise-ceiling-normalized encoding accuracy to 100 for vertices where
	# the the correlation is higher than the noise ceiling, to prevent encoding
	# accuracy values higher than 100%
	lh_explained_variance_nsdcore_test_id_sub\
		[lh_explained_variance_nsdcore_test_id_sub>100] = 100
	rh_explained_variance_nsdcore_test_id_sub\
		[rh_explained_variance_nsdcore_test_id_sub>100] = 100
	lh_explained_variance_nsdcore_test_ood_sub\
		[lh_explained_variance_nsdcore_test_ood_sub>100] = 100
	rh_explained_variance_nsdcore_test_ood_sub\
		[rh_explained_variance_nsdcore_test_ood_sub>100] = 100
	lh_explained_variance_nsdsynthetic_sub\
		[lh_explained_variance_nsdsynthetic_sub>100] = 100
	rh_explained_variance_nsdsynthetic_sub\
		[rh_explained_variance_nsdsynthetic_sub>100] = 100

	# Store the encoding accuracy results
	lh_explained_variance_nsdcore_test_id.append(
		lh_explained_variance_nsdcore_test_id_sub)
	rh_explained_variance_nsdcore_test_id.append(
		rh_explained_variance_nsdcore_test_id_sub)
	lh_explained_variance_nsdcore_test_ood.append(
		lh_explained_variance_nsdcore_test_ood_sub)
	rh_explained_variance_nsdcore_test_ood.append(
		rh_explained_variance_nsdcore_test_ood_sub)
	lh_explained_variance_nsdsynthetic.append(
		lh_explained_variance_nsdsynthetic_sub)
	rh_explained_variance_nsdsynthetic.append(
		rh_explained_variance_nsdsynthetic_sub)

	# Delete unused variables
	del lh_correlation_nsdcore_test_id_sub, rh_correlation_nsdcore_test_id_sub, \
		lh_r2_nsdcore_test_id_sub, rh_r2_nsdcore_test_id_sub, \
		lh_nc_nsdcore_test_id_sub, rh_nc_nsdcore_test_id_sub, \
		lh_explained_variance_nsdcore_test_id_sub, \
		rh_explained_variance_nsdcore_test_id_sub, \
		lh_correlation_nsdcore_test_ood_sub, rh_correlation_nsdcore_test_ood_sub, \
		lh_r2_nsdcore_test_ood_sub, rh_r2_nsdcore_test_ood_sub, \
		lh_nc_nsdcore_test_ood_sub, rh_nc_nsdcore_test_ood_sub, \
		lh_explained_variance_nsdcore_test_ood_sub, \
		rh_explained_variance_nsdcore_test_ood_sub, \
		lh_correlation_nsdsynthetic_sub, rh_correlation_nsdsynthetic_sub, \
		lh_r2_nsdsynthetic_sub, rh_r2_nsdsynthetic_sub, \
		lh_nc_nsdsynthetic_sub, rh_nc_nsdsynthetic_sub, \
		lh_explained_variance_nsdsynthetic_sub, \
		rh_explained_variance_nsdsynthetic_sub


# =============================================================================
# Save the prediction accuracy
# =============================================================================
results = {
	'lh_correlation_nsdcore_test_id': lh_correlation_nsdcore_test_id,
	'rh_correlation_nsdcore_test_id': rh_correlation_nsdcore_test_id,
	'lh_r2_nsdcore_test_id': lh_r2_nsdcore_test_id,
	'rh_r2_nsdcore_test_id': rh_r2_nsdcore_test_id,
	'lh_nc_nsdcore_test_id': lh_nc_nsdcore_test_id,
	'rh_nc_nsdcore_test_id': rh_nc_nsdcore_test_id,
	'lh_explained_variance_nsdcore_test_id' : lh_explained_variance_nsdcore_test_id,
	'rh_explained_variance_nsdcore_test_id': rh_explained_variance_nsdcore_test_id,
	'lh_correlation_nsdcore_test_ood': lh_correlation_nsdcore_test_ood,
	'rh_correlation_nsdcore_test_ood': rh_correlation_nsdcore_test_ood,
	'lh_r2_nsdcore_test_ood': lh_r2_nsdcore_test_ood,
	'rh_r2_nsdcore_test_ood': rh_r2_nsdcore_test_ood,
	'lh_nc_nsdcore_test_ood': lh_nc_nsdcore_test_ood,
	'rh_nc_nsdcore_test_ood': rh_nc_nsdcore_test_ood,
	'lh_explained_variance_nsdcore_test_ood' : lh_explained_variance_nsdcore_test_ood,
	'rh_explained_variance_nsdcore_test_ood': rh_explained_variance_nsdcore_test_ood,
	'lh_correlation_nsdsynthetic' : lh_correlation_nsdsynthetic,
	'rh_correlation_nsdsynthetic': rh_correlation_nsdsynthetic,
	'lh_r2_nsdsynthetic' : lh_r2_nsdsynthetic,
	'rh_r2_nsdsynthetic': rh_r2_nsdsynthetic,
	'lh_nc_nsdsynthetic' : lh_nc_nsdsynthetic,
	'rh_nc_nsdsynthetic': rh_nc_nsdsynthetic,
	'lh_explained_variance_nsdsynthetic' : lh_explained_variance_nsdsynthetic,
	'rh_explained_variance_nsdsynthetic': rh_explained_variance_nsdsynthetic
	}

save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'encoding_accuracy', 'zscore-'+str(args.zscore), 'model-'+args.model)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)
