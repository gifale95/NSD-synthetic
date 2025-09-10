"""Test the image condition identification accuracy of the encoding model's in
silico neural responses for NSD-core's ID and OOD images, and for
NSD-synthetic's images.

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
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Image condition identification <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Empty result lists
# =============================================================================
correlation_matrix = []
rank_nsdcoreid = []
rank_nsdcoreood = []
rank_nsdsynthetic = []


# =============================================================================
# Loop across subjects
# =============================================================================
for s, sub in enumerate(tqdm(args.subjects)):


# =============================================================================
# Get indices of vertices with ncsnr above threshold
# =============================================================================
	# NSD-synthetic
	metadata_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+format(sub),
		'meatadata_nsdsynthetic.npy')
	metadata = np.load(metadata_dir, allow_pickle=True).item()
	lh_ncsnr_nsdsynthetic = metadata['lh_ncsnr']
	rh_ncsnr_nsdsynthetic = metadata['rh_ncsnr']
	lh_idx_nsdsynthetic = lh_ncsnr_nsdsynthetic > args.ncsnr_threshold
	rh_idx_nsdsynthetic = rh_ncsnr_nsdsynthetic > args.ncsnr_threshold

	# NSD-core
	metadata_dir = os.path.join(args.project_dir, 'results',
		'nsdcore_id_ood_tests', 'fmri_betas', 'zscore-'+str(args.zscore),
		'sub-0'+str(sub), 'meatadata_nsdcore.npy')
	metadata = np.load(metadata_dir, allow_pickle=True).item()
	lh_ncsnr_nsdcore = metadata['lh_ncsnr']
	rh_ncsnr_nsdcore = metadata['rh_ncsnr']
	lh_idx_nsdcore = lh_ncsnr_nsdcore > args.ncsnr_threshold
	rh_idx_nsdcore = rh_ncsnr_nsdcore > args.ncsnr_threshold

	# Only retain vertices with nscnr above threshold for both NSD-synthetic
	# and NSD-core
	lh_idx = np.where(np.logical_and(lh_idx_nsdsynthetic, lh_idx_nsdcore))[0]
	rh_idx = np.where(np.logical_and(rh_idx_nsdsynthetic, rh_idx_nsdcore))[0]


# =============================================================================
# Load the recorded and predicted fMRI responses for the NSD-core test images
# =============================================================================
	# Recorded fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+str(sub))
	lh_betas_nsdcore_test_id = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdcore_test_id.h5'), 'r')['betas'][:,lh_idx]
	rh_betas_nsdcore_test_id = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdcore_test_id.h5'), 'r')['betas'][:,rh_idx]
	lh_betas_nsdcore_test_ood = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdcore_test_ood.h5'), 'r')['betas'][:,lh_idx]
	rh_betas_nsdcore_test_ood = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdcore_test_ood.h5'), 'r')['betas'][:,rh_idx]

	# Predicted fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'predicted_fmri', 'zscore-'+str(args.zscore), 'model-'+args.model,
		'predicted_fmri_sub-0'+str(sub)+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	lh_betas_nsdcore_test_id_pred = data['lh_betas_nsdcore_test_id_pred'][:,lh_idx]
	rh_betas_nsdcore_test_id_pred = data['rh_betas_nsdcore_test_id_pred'][:,rh_idx]
	lh_betas_nsdcore_test_ood_pred = data['lh_betas_nsdcore_test_ood_pred'][:,lh_idx]
	rh_betas_nsdcore_test_ood_pred = data['rh_betas_nsdcore_test_ood_pred'][:,rh_idx]


# =============================================================================
# Load the recorded and predicted fMRI responses for the NSD-synthetic images
# =============================================================================
	# Recorded fMRI
	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+str(sub))
	lh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'lh_betas_nsdsynthetic.h5'), 'r')['betas'][:,lh_idx]
	rh_betas_nsdsynthetic = h5py.File(os.path.join(data_dir,
		'rh_betas_nsdsynthetic.h5'), 'r')['betas'][:,rh_idx]

	# Predicted fMRI
	lh_betas_nsdsynthetic_pred = data['lh_betas_nsdsynthetic_pred'][:,lh_idx]
	rh_betas_nsdsynthetic_pred = data['rh_betas_nsdsynthetic_pred'][:,rh_idx]
	del data


# =============================================================================
# Append the fMRI responses across vertices and test splits
# =============================================================================
	# Append the fMRI responses across vertices
	betas_nsdcore_test_id = np.append(lh_betas_nsdcore_test_id,
		rh_betas_nsdcore_test_id, 1)
	betas_nsdcore_test_ood = np.append(lh_betas_nsdcore_test_ood,
		rh_betas_nsdcore_test_ood, 1)
	betas_nsdsynthetic = np.append(lh_betas_nsdsynthetic,
		rh_betas_nsdsynthetic, 1)
	betas_nsdcore_test_id_pred = np.append(lh_betas_nsdcore_test_id_pred,
		rh_betas_nsdcore_test_id_pred, 1)
	betas_nsdcore_test_ood_pred = np.append(lh_betas_nsdcore_test_ood_pred,
		rh_betas_nsdcore_test_ood_pred, 1)
	betas_nsdsynthetic_pred = np.append(lh_betas_nsdsynthetic_pred,
		rh_betas_nsdsynthetic_pred, 1)

	# Append the fMRI responses across test splits
	betas = np.append(betas_nsdcore_test_id, betas_nsdcore_test_ood, 0)
	betas = np.append(betas, betas_nsdsynthetic, 0).astype(np.float32)
	betas_pred = np.append(betas_nsdcore_test_id_pred,
		betas_nsdcore_test_ood_pred, 0)
	betas_pred = np.append(betas_pred, betas_nsdsynthetic_pred,
		0).astype(np.float32)

	# Delete unused variables
	del lh_betas_nsdcore_test_id, rh_betas_nsdcore_test_id, \
		lh_betas_nsdcore_test_ood, rh_betas_nsdcore_test_ood, \
		lh_betas_nsdsynthetic, rh_betas_nsdsynthetic, \
		lh_betas_nsdcore_test_id_pred, rh_betas_nsdcore_test_id_pred, \
		lh_betas_nsdcore_test_ood_pred, rh_betas_nsdcore_test_ood_pred, \
		lh_betas_nsdsynthetic_pred, rh_betas_nsdsynthetic_pred, \
		betas_nsdcore_test_id, betas_nsdcore_test_ood, betas_nsdsynthetic, \
		betas_nsdcore_test_id_pred, betas_nsdcore_test_ood_pred, \
		betas_nsdsynthetic_pred


# =============================================================================
# Correlate the recorded and predicted fMRI responses of each image condition
# =============================================================================
	correlation_matrix_sub = np.zeros((len(betas), len(betas)),
		dtype=np.float32)

	for i1 in range(len(betas)):
		for i2 in range(len(betas)):
			correlation_matrix_sub[i1,i2] = pearsonr(betas[i1],
				betas_pred[i2])[0]

	correlation_matrix.append(correlation_matrix_sub)


# =============================================================================
# Get the identification rank of each image condition
# =============================================================================
	img_per_split = 284
	rank_nsdcoreid_sub = np.zeros(img_per_split, dtype=np.int32)
	rank_nsdcoreood_sub = np.zeros(img_per_split, dtype=np.int32)
	rank_nsdsynthetic_sub = np.zeros(img_per_split, dtype=np.int32)

	# Split the correlation matrix into test splits
	correlation_matrix_nsdcoreid = correlation_matrix_sub[:284,:284]
	correlation_matrix_nsdcoreood = correlation_matrix_sub[284:568,284:568]
	correlation_matrix_nsdsynthetic = correlation_matrix_sub[568:,568:]

	# Get the identification rank of each image condition
	for i in range(len(correlation_matrix_nsdcoreid)):
		rank_nsdcoreid_sub[i] = np.where(np.argsort(
			correlation_matrix_nsdcoreid[i])[::-1] == i)[0][0]
		rank_nsdcoreood_sub[i] = np.where(np.argsort(
			correlation_matrix_nsdcoreood[i])[::-1] == i)[0][0]
		rank_nsdsynthetic_sub[i] = np.where(np.argsort(
			correlation_matrix_nsdsynthetic[i])[::-1] == i)[0][0]
	del correlation_matrix_sub

	# Store the identification ranks
	rank_nsdcoreid.append(rank_nsdcoreid_sub)
	rank_nsdcoreood.append(rank_nsdcoreood_sub)
	rank_nsdsynthetic.append(rank_nsdsynthetic_sub)
	del rank_nsdcoreid_sub, rank_nsdcoreood_sub, rank_nsdsynthetic_sub


# =============================================================================
# Format the result lists into numpy arrays
# =============================================================================
correlation_matrix = np.asarray(correlation_matrix)
rank_nsdcoreid = np.asarray(rank_nsdcoreid)
rank_nsdcoreood = np.asarray(rank_nsdcoreood)
rank_nsdsynthetic = np.asarray(rank_nsdsynthetic)


# =============================================================================
# Save the prediction accuracy
# =============================================================================
results = {
	'correlation_matrix': correlation_matrix,
	'rank_nsdcoreid': rank_nsdcoreid,
	'rank_nsdcoreood': rank_nsdcoreood,
	'rank_nsdsynthetic': rank_nsdsynthetic
	}

save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'image_identification_accuracy', 'zscore-'+str(args.zscore),
	'model-'+args.model)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'image_identification_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)
