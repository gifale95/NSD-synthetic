"""Train linearizing encoding models using the NSD-core subject-unique image
features and fMRI responses, and use them to predict fMRI responses for the:
(i) NSD-core's test images.
(ii) NSD-synthetic 284 images.

Parameters
----------
subject : int
	Number of the used NSD subject.
train_test_session_control : int
	If '1', use the train and test splits consist of image conditions from
	non-overlapping fMRI scan sessions.
tot_vertex_splits : int
	If regression='ridge', total amount of fMRI vertex splits.
vertex_split : int
	If regression='ridge', vertex split used.
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
import h5py
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--train_test_session_control', type=int, default=0)
parser.add_argument('--tot_vertex_splits', type=int, default=14)
parser.add_argument('--vertex_split', type=int, default=0)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--layer', default='all', type=str)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Train encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Load the metadata
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results',
	'train_test_session_control-'+str(args.train_test_session_control),
	'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+format(args.subject),
	'meatadata_nsdcore.npy')

metadata_nsdcore = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Load the image features
# =============================================================================
# Load the image features
if args.regression == 'linear':

	# Load the PCA-downsampled images features
	features_dir = os.path.join(args.project_dir, 'results',
		'train_test_session_control-'+str(args.train_test_session_control),
		'image_features', 'pca_features', 'model-'+args.model, 'layer-'+
		args.layer, 'pca_features_sub-0'+str(args.subject)+'.npy')
	features = np.load(features_dir, allow_pickle=True).item()
	features_nsdcore_train = features['features_nsdcore_train']
	features_nsdcore_test = features['features_nsdcore_test']
	features_nsdsynthetic = features['features_nsdsynthetic']

elif args.regression == 'ridge':

	if args.layer == 'all':
		pass
	else:
		# Load the full images features (NSD-core train)
		features_dir = os.path.join(args.project_dir, 'results',
			'image_features', 'full_features', 'model-'+args.model, 'nsdcore')
		features_nsdcore_train = []
		for i in tqdm(metadata_nsdcore['train_img_num']):
			file_dir = os.path.join(features_dir, 'img_'+format(i,'06')+'.npy')
			features_nsdcore_train.append(np.load(file_dir,
				allow_pickle=True).item()[args.layer].flatten())
		features_nsdcore_train = np.asarray(features_nsdcore_train)
		# Load the full images features (NSD-core test)
		features_nsdcore_test = []
		for i in tqdm(metadata_nsdcore['test_img_num']):
			file_dir = os.path.join(features_dir, 'img_'+format(i,'06')+'.npy')
			features_nsdcore_test.append(np.load(file_dir,
				allow_pickle=True).item()[args.layer].flatten())
		features_nsdcore_test = np.asarray(features_nsdcore_test)
		# Load the full images features (NSD-synthetic stimuli)
		features_nsdsynthetic = []
		features_dir = os.path.join(args.project_dir, 'results', 'image_features',
			'full_features', 'model-'+args.model, 'nsdsynthetic_stimuli')
		fmaps_list = os.listdir(features_dir)
		fmaps_list.sort()
		for i in tqdm(fmaps_list):
			file_dir = os.path.join(features_dir, i)
			features_nsdsynthetic.append(np.load(file_dir,
				allow_pickle=True).item()[args.layer].flatten())
		# Load the full images features (NSD-synthetic colorstimuli)
		fmaps_dir = os.path.join(args.project_dir, 'results', 'image_features',
			'full_features', 'model-'+args.model, 'nsdsynthetic_colorstimuli_subj0'+
			str(args.subject))
		fmaps_list = os.listdir(fmaps_dir)
		fmaps_list.sort()
		for i in tqdm(fmaps_list):
			file_dir = os.path.join(features_dir, i)
			features_nsdsynthetic.append(np.load(file_dir,
				allow_pickle=True).item()[args.layer].flatten())
		features_nsdsynthetic = np.asarray(features_nsdsynthetic)


# =============================================================================
# Load the fMRI responses
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results',
	'train_test_session_control-'+str(args.train_test_session_control),
	'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+format(args.subject))
lh_betas_nsdcore_train = h5py.File(os.path.join(data_dir,
	'lh_betas_nsdcore_train.h5'), 'r')['betas'][:]
rh_betas_nsdcore_train = h5py.File(os.path.join(data_dir,
	'rh_betas_nsdcore_train.h5'), 'r')['betas'][:]

# Split the vertices
if args.regression == 'ridge':
	# Get the vertex split indices
	n_vertices = lh_betas_nsdcore_train.shape[1]
	vertices_per_split = int(np.ceil(n_vertices / args.tot_vertex_splits))
	idx_start = args.vertex_split * vertices_per_split
	idx_end = idx_start + vertices_per_split
	# Select the vertices from the chosen split
	lh_betas_nsdcore_train = lh_betas_nsdcore_train[:,idx_start:idx_end]
	rh_betas_nsdcore_train = rh_betas_nsdcore_train[:,idx_start:idx_end]


# =============================================================================
# Train the encoding models, and predict fMRI responses for the test images
# =============================================================================
# For each fMRI vertex, train a linear regression using (i) its fMRI responses
# for the NSD-core subject-unique images as the criterion, and (ii) the
# corresponding image features as the predictor. Then, use the trained weights
# to predict the vertex responses for the NSD-core shared 1,000 images and the
# 284 NSD-synthetic images.

# Train encoding models using the NSD-core subject-unique images: fit the
# regression models at each fMRI vertex
if args.regression == 'linear':

	lh_reg = LinearRegression().fit(features_nsdcore_train,
		lh_betas_nsdcore_train)
	rh_reg = LinearRegression().fit(features_nsdcore_train,
		rh_betas_nsdcore_train)

elif args.regression == 'ridge':

	alphas = np.asarray((1000000, 100000, 10000, 1000, 100, 10, 1, 0.1, 0.01,
		0.001))
	lh_reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
	rh_reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
	lh_reg.fit(features_nsdcore_train, lh_betas_nsdcore_train)
	rh_reg.fit(features_nsdcore_train, rh_betas_nsdcore_train)

# Use the learned weights to predict fMRI responses
lh_betas_nsdcore_test_pred = lh_reg.predict(features_nsdcore_test)
rh_betas_nsdcore_test_pred = rh_reg.predict(features_nsdcore_test)
lh_betas_nsdsynthetic_pred = lh_reg.predict(features_nsdsynthetic)
rh_betas_nsdsynthetic_pred = rh_reg.predict(features_nsdsynthetic)


# =============================================================================
# Save the predicted betas
# =============================================================================
predicted_fmri = {
	'lh_betas_nsdcore_test_pred': lh_betas_nsdcore_test_pred,
	'rh_betas_nsdcore_test_pred': rh_betas_nsdcore_test_pred,
	'lh_betas_nsdsynthetic_pred': lh_betas_nsdsynthetic_pred,
	'rh_betas_nsdsynthetic_pred': rh_betas_nsdsynthetic_pred
	}

save_dir = os.path.join(args.project_dir, 'results',
	'train_test_session_control-'+str(args.train_test_session_control),
	'predicted_fmri', 'zscore-'+str(args.zscore), 'model-'+args.model,
	'layer-'+args.layer, 'regression-'+args.regression)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.regression == 'linear':
	file_name = 'predicted_fmri_sub-0' + str(args.subject) + '.npy'
elif args.regression == 'ridge':
	file_name = 'predicted_fmri_sub-0' + str(args.subject) + \
		'_vertex_split-' + format(args.vertex_split, '03') + '.npy'

np.save(os.path.join(save_dir, file_name), predicted_fmri)
