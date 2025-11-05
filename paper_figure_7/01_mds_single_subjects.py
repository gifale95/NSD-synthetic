"""Perform MDS on the trial-average fMRI responses, or on the DDN features, for
the NSD-core images.

Parameters
----------
subject : int
	Number of the used subject.
data_ood_selection : str
	If 'fmri', perform MDS on the fMRI responses for the NSD-core images.
	If 'dnn', perform MDS on the DNN features for the NSD-core images.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept vertices: only vertices above this
	threshold are used.
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet', 'resnet50', 'moco', and 'vit_b_32'.
layer : str
	If 'all', apply PCA on the features from all model layers. If a layer name
	is given, PCA is applied on the features of that layer.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from sklearn.manifold import MDS
import h5py
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--data_ood_selection', default='fmri', type=str)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--model', default='vit_b_32', type=str)
parser.add_argument('--layer', default='all', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()

print('>>> MDS single subjects <<<')
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
# Load the fMRI responses
# =============================================================================
if args.data_ood_selection == 'fmri':

	# Load the metadata
	data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+format(args.subject))
	metadata = np.load(os.path.join(data_dir, 'metadata_nsdcore.npy'),
		allow_pickle=True).item()
	# Get indices of vertices with ncsnr above threshold
	lh_ncsnr = metadata['lh_ncsnr']
	rh_ncsnr = metadata['rh_ncsnr']
	# Select the above-threshold vertices
	lh_idx = np.where(lh_ncsnr > args.ncsnr_threshold)[0]
	rh_idx = np.where(rh_ncsnr > args.ncsnr_threshold)[0]

	# Load the trial-average betas
	lh = h5py.File(os.path.join(data_dir, 'lh_betas_nsdcore.h5'),
		'r')['betas'][:,lh_idx]
	rh = h5py.File(os.path.join(data_dir, 'rh_betas_nsdcore.h5'),
		'r')['betas'][:,rh_idx]
	# Append the data from left and right hemispheres
	data = np.append(lh, rh, 1)
	del lh, rh

# =============================================================================
# Load the DNN responses
# =============================================================================
if args.data_ood_selection == 'dnn':

	# Load the DNN responses
	data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'image_features', 'pca_features', 'model-'+args.model, 'layer-'+
		args.layer, 'pca_features_sub-0'+str(args.subject)+'.npy')
	data = np.load(data_dir)


# =============================================================================
# Apply MDS
# =============================================================================
embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=20200220)

data_mds = embedding.fit_transform(data)


# =============================================================================
# Save the MDS results
# =============================================================================
results = {
	'img_num': img_num,
	'data_mds': data_mds
	}

if args.data_ood_selection == 'fmri':
	save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'mds_single_subjects', 'data_ood_selection-'+args.data_ood_selection,
		'zscore-'+str(args.zscore))
elif args.data_ood_selection == 'dnn':
	save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
		'mds_single_subjects', 'data_ood_selection-'+args.data_ood_selection,
		'model-'+args.model, 'layer-'+args.layer)

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

file_name = 'mds_subject-' + format(args.subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), results)
