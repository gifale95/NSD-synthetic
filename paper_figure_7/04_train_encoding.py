"""Train linearizing encoding models using the NSD-core subject-unique image
features and fMRI responses, and use them to predict fMRI responses for the:
(i) NSD-core ID test images.
(i) NSD-core OOD test images.
(iii) NSD-synthetic 284 images.

Parameters
----------
subject : int
	Number of the used NSD subject.
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
import h5py
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Train encoding models <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the image features and fMRI responses
# =============================================================================
# Load the PCA-downsampled image features
features_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'pca_features', 'model-'+args.model, 'layer-all', 'pca_features_sub-0'+
	str(args.subject)+'.npy')
features = np.load(features_dir, allow_pickle=True).item()
features_nsdcore_train = features['features_nsdcore_train']
features_nsdcore_test_id = features['features_nsdcore_test_id']
features_nsdcore_test_ood = features['features_nsdcore_test_ood']
features_nsdsynthetic = features['features_nsdsynthetic']

# Load the fMRI responses
data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'fmri_betas', 'zscore-'+str(args.zscore), 'sub-0'+format(args.subject))
lh_betas_nsdcore_train = h5py.File(os.path.join(data_dir,
	'lh_betas_nsdcore_train.h5'), 'r')['betas'][:]
rh_betas_nsdcore_train = h5py.File(os.path.join(data_dir,
	'rh_betas_nsdcore_train.h5'), 'r')['betas'][:]


# =============================================================================
# Train the encoding models, and predict fMRI responses for the test images
# =============================================================================
# For each fMRI vertex, train a linear regression using the fMRI responses
# for the NSD-core training images as the criterion, and the corresponding image
# features as the predictor. Then, use the trained weights to predict the vertex
# responses for the NSD-core ID and OOD test images, and for the 284
# NSD-synthetic images.

# Train encoding models using the NSD-core subject-unique images: fit the
# regression models at each fMRI vertex
lh_reg = LinearRegression().fit(features_nsdcore_train, lh_betas_nsdcore_train)
rh_reg = LinearRegression().fit(features_nsdcore_train, rh_betas_nsdcore_train)

# Use the learned weights to predict fMRI responses
lh_betas_nsdcore_test_id_pred = lh_reg.predict(features_nsdcore_test_id)
rh_betas_nsdcore_test_id_pred = rh_reg.predict(features_nsdcore_test_id)
lh_betas_nsdcore_test_ood_pred = lh_reg.predict(features_nsdcore_test_ood)
rh_betas_nsdcore_test_ood_pred = rh_reg.predict(features_nsdcore_test_ood)
lh_betas_nsdsynthetic_pred = lh_reg.predict(features_nsdsynthetic)
rh_betas_nsdsynthetic_pred = rh_reg.predict(features_nsdsynthetic)


# =============================================================================
# Save the predicted betas
# =============================================================================
predicted_fmri = {
	'lh_betas_nsdcore_test_id_pred': lh_betas_nsdcore_test_id_pred,
	'rh_betas_nsdcore_test_id_pred': rh_betas_nsdcore_test_id_pred,
	'lh_betas_nsdcore_test_ood_pred': lh_betas_nsdcore_test_ood_pred,
	'rh_betas_nsdcore_test_ood_pred': rh_betas_nsdcore_test_ood_pred,
	'lh_betas_nsdsynthetic_pred': lh_betas_nsdsynthetic_pred,
	'rh_betas_nsdsynthetic_pred': rh_betas_nsdsynthetic_pred
	}

save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'predicted_fmri', 'zscore-'+str(args.zscore), 'model-'+args.model)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'predicted_fmri_sub-0' + str(args.subject) + '.npy'

np.save(os.path.join(save_dir, file_name), predicted_fmri)
