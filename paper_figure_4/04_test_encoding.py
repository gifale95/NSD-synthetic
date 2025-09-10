"""Test the encoding model's prediction accuracy in-distribution (using 284
NSD-core's shared images), and out-of-distribution (using NSD-synthetic's 284
images).

Parameters
----------
subjects : list
	List of the used NSD subjects.
train_test_session_control : int
	If '1', use the train and test splits consist of image conditions from
	non-overlapping fMRI scan sessions.
tot_vertex_splits : int
	If regression='ridge', total amount of fMRI vertex splits.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet', 'resnet50', 'moco', and 'vit_b_32'.
layer : str
	If 'all', train the encoding models on the features from all model layers.
	If a layer name is given, the encoding models are trained on the features of
	that layer.
regression : str
	If 'linear', the encoding models will consist of linear regressions that
	predict fMRI responses using PCA-downsampled image features as predictors.
	If 'ridge', the encoding models will consist of ridge regressions that
	predict fMRI responses using full image features as predictors.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--train_test_session_control', type=int, default=0)
parser.add_argument('--tot_vertex_splits', type=int, default=14)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--layer', default='all', type=str)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Test encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


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
	data_dir = os.path.join(args.project_dir, 'results',
		'train_test_session_control-'+str(args.train_test_session_control),
		'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+format(sub),
		'meatadata_nsdcore.npy')
	metadata_nsdcore = np.load(data_dir, allow_pickle=True).item()

	# Recorded fMRI
	data_dir = os.path.join(args.project_dir, 'results',
		'train_test_session_control-'+str(args.train_test_session_control),
		'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+format(sub))
	lh_betas_nsdcore_test = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdcore_test.h5'), 'r')['betas'][:]
	rh_betas_nsdcore_test = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdcore_test.h5'), 'r')['betas'][:]

	# Predicted fMRI
	if args.regression == 'linear':
		data_dir = os.path.join(args.project_dir, 'results',
			'train_test_session_control-'+str(args.train_test_session_control),
			'predicted_fmri', 'zscore-'+str(args.zscore), 'model-'+args.model,
			'layer-'+args.layer, 'regression-'+args.regression,
			'predicted_fmri_sub-0'+str(sub)+'.npy')
		data = np.load(data_dir, allow_pickle=True).item()
		lh_betas_nsdcore_test_pred = data['lh_betas_nsdcore_test_pred']
		rh_betas_nsdcore_test_pred = data['rh_betas_nsdcore_test_pred']
	elif args.regression == 'ridge':
		for vs in range(args.tot_vertex_splits):
			data_dir = os.path.join(args.project_dir, 'results',
				'train_test_session_control-'+str(args.train_test_session_control),
				'predicted_fmri', 'zscore-'+str(args.zscore), 'model-'+args.model,
				'layer-'+args.layer, 'regression-'+args.regression,
				'predicted_fmri_sub-0'+str(sub)+'_vertex_split-'+
				format(vs, '03')+'.npy')
			data = np.load(data_dir, allow_pickle=True).item()
			if vs == 0:
				lh_betas_nsdcore_test_pred = data['lh_betas_nsdcore_test_pred']
				rh_betas_nsdcore_test_pred = data['rh_betas_nsdcore_test_pred']
			else:
				lh_betas_nsdcore_test_pred = np.append(
					lh_betas_nsdcore_test_pred,
					data['lh_betas_nsdcore_test_pred'], 1)
				rh_betas_nsdcore_test_pred = np.append(
					rh_betas_nsdcore_test_pred,
					data['rh_betas_nsdcore_test_pred'], 1)
			del data

	# ncsnr (284 image conditions)
	lh_ncsnr_nsdcore_test = metadata_nsdcore['lh_ncsnr_284']
	rh_ncsnr_nsdcore_test = metadata_nsdcore['rh_ncsnr_284']

	# Convert the ncsnr to noise ceiling (284 image conditions)
	test_img_repeats = metadata_nsdcore['test_img_repeats']
	img_reps_1 = len(np.where(test_img_repeats == 1)[0])
	img_reps_2 = len(np.where(test_img_repeats == 2)[0])
	img_reps_3 = len(np.where(test_img_repeats == 3)[0])
	norm_term = (img_reps_1/1 + img_reps_2/2 + img_reps_3/3) / \
		(img_reps_1 + img_reps_2 + img_reps_3)
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
		'zscore-'+str(args.zscore), 'sub-0'+format(sub))
	lh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:]
	rh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:]

	# Predicted fMRI
	if args.regression == 'linear':
		lh_betas_nsdsynthetic_pred = data['lh_betas_nsdsynthetic_pred']
		rh_betas_nsdsynthetic_pred = data['rh_betas_nsdsynthetic_pred']
	elif args.regression == 'ridge':
		for vs in range(args.tot_vertex_splits):
			data_dir = os.path.join(args.project_dir, 'results',
				'train_test_session_control-'+str(args.train_test_session_control),
				'predicted_fmri', 'zscore-'+str(args.zscore), 'model-'+args.model,
				'layer-'+args.layer, 'regression-'+args.regression,
				'predicted_fmri_sub-0'+str(sub)+'_vertex_split-'+
				format(vs, '03')+'.npy')
			data = np.load(data_dir, allow_pickle=True).item()
			if vs == 0:
				lh_betas_nsdsynthetic_pred = data['lh_betas_nsdsynthetic_pred']
				rh_betas_nsdsynthetic_pred = data['rh_betas_nsdsynthetic_pred']
			else:
				lh_betas_nsdsynthetic_pred = np.append(
					lh_betas_nsdsynthetic_pred,
					data['lh_betas_nsdsynthetic_pred'], 1)
				rh_betas_nsdsynthetic_pred = np.append(
					rh_betas_nsdsynthetic_pred,
					data['rh_betas_nsdsynthetic_pred'], 1)
			del data

	# ncsnr
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+format(sub),
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

	# Set the noise-ceiling-normalized encoding accuracy to 100 for vertices where
	# the the correlation is higher than the noise ceiling, to prevent encoding
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

save_dir = os.path.join(args.project_dir, 'results',
	'train_test_session_control-'+str(args.train_test_session_control),
	'encoding_accuracy', 'zscore-'+str(args.zscore), 'model-'+args.model,
	'layer-'+args.layer, 'regression-'+args.regression)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)
