"""Compute the fMRI response distance (in MDS space) between the different image
classes.

Parameters
----------
subjects : list
	List of the used NSD subjects.
data_ood_selection : str
	If 'fmri', the ID/OD splits are defined based on fMRI responses.
	If 'dnn', the ID/OD splits are defined based on DNN features.
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

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--data_ood_selection', default='fmri', type=str)
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> MDS distance scores <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the single subject MDS results
# =============================================================================
betas_mds_single_sub = []

for s, sub in enumerate(args.subjects):

	data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'mds_all_subjects', 'data_ood_selection-'+args.data_ood_selection,
		'mds_subject-'+str(sub)+'.npy')

	betas_mds_single_sub.append(np.load(data_dir))


# =============================================================================
# Load the subject average
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'mds_all_subjects', 'data_ood_selection-'+args.data_ood_selection,
	'mds_subject-all.npy')

betas_mds_all_sub = np.load(data_dir)


# =============================================================================
# Compute the euclidean distance between NSD-core and NSD-synthetic responses
# =============================================================================
# Loop across image classes
test_split_euclidean_distance = {}
test_splits = ['nsdcore_id', 'nsdcore_ood', 'nsd_synthetic']
for ts in tqdm(test_splits):

	# Get the indices of the images from the selected test split
	if ts == 'nsdcore_id':
		test_img_idx = np.arange(284)
	elif ts == 'nsdcore_ood':
		test_img_idx = np.arange(284, 568)
	elif ts == 'nsd_synthetic':
		test_img_idx = np.arange(568, 852)

	# Loop over subjects
	distance = []
	for s, betas_sub in enumerate(betas_mds_single_sub):

		# Get the indices of the NSD-core training images
		train_img_idx = np.arange(852, len(betas_sub))

		# Compute euclidean distance between NSD-core train and the test
		# responses
		distance_sub = np.zeros((len(test_img_idx), len(train_img_idx)),
			dtype=np.float32)
		for i1, idx_1 in enumerate(test_img_idx):
			for i2, idx_2 in enumerate(train_img_idx):
				distance_sub[i1,i2] = np.sum(np.sqrt(np.square(
					betas_sub[idx_1] - betas_sub[idx_2])))

		# Average the euclidean distances across all images withing the same
		# NSD-synthetic image class and across all NSD-core training images
		distance_sub = np.mean(distance_sub)

		# Append the distance scores across subjects
		distance.append(distance_sub)
		del distance_sub

	# Store the distance scores for each NSD-synthetic image class
	test_split_euclidean_distance[ts] = np.asarray(distance)


# =============================================================================
# Save the MDS results
# =============================================================================
results = {
	'betas_mds_single_sub': betas_mds_single_sub,
	'betas_mds_all_sub': betas_mds_all_sub,
	'test_split_euclidean_distance': test_split_euclidean_distance
	}

save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'mds_all_subjects', 'data_ood_selection-'+args.data_ood_selection)

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

np.save(os.path.join(save_dir, 'mds.npy'), results)
