"""Downsample the image features with PCA.

Parameters
----------
subject : int
	Number of the used NSD subject.
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet' and 'vit_b_32'.
n_components : int
	Number of PCA components retained.
project_dir : str
	Directory of the project folder.

"""

import argparse
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--n_components', default=250, type=int)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Downsample image features with PCA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the metadata
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas', 'sub-0'+
	format(args.subject))

metadata_nsdcore = np.load(os.path.join(data_dir, 'meatadata_nsdcore.npy'),
	allow_pickle=True).item()


# =============================================================================
# NSD-core train image features
# =============================================================================
features_dir = os.path.join(args.project_dir, 'results', 'image_features',
	'full_features', 'model-'+args.model, 'nsdcore')

features_nsdcore_train = []
for i in tqdm(metadata_nsdcore['train_img_num']):
	features_nsdcore_train.append(np.load(os.path.join(
		features_dir, 'img_'+format(i,'06')+'.npy')))
features_nsdcore_train = np.asarray(features_nsdcore_train)

# Z-score the image features
scaler = StandardScaler()
scaler.fit(features_nsdcore_train)
features_nsdcore_train = scaler.transform(features_nsdcore_train)

# Downsample the features with PCA
pca = PCA(n_components=args.n_components, random_state=20200220)
pca.fit(features_nsdcore_train)
features_nsdcore_train = pca.transform(features_nsdcore_train)


# =============================================================================
# NSD-core test image features
# =============================================================================
features_nsdcore_test = []
for i in tqdm(metadata_nsdcore['test_img_num']):
	features_nsdcore_test.append(np.load(os.path.join(
		features_dir, 'img_'+format(i,'06')+'.npy')))
features_nsdcore_test = np.asarray(features_nsdcore_test)

# Z-score the image features
features_nsdcore_test = scaler.transform(features_nsdcore_test)

# Downsample the features with PCA
features_nsdcore_test = pca.transform(features_nsdcore_test)


# =============================================================================
# NSD-synthetic image features
# =============================================================================
features_nsdsynthetic = []

# Load the NSD-synthetic stimuli features
features_dir = os.path.join(args.project_dir, 'results', 'image_features',
	'full_features', 'model-'+args.model, 'nsdsynthetic_stimuli')
fmaps_list = os.listdir(features_dir)
fmaps_list.sort()
for i in tqdm(fmaps_list):
	features_nsdsynthetic.append(np.load(os.path.join(features_dir, i)))

# Load the NSD-synthetic colorstimuli features
fmaps_dir = os.path.join(args.project_dir, 'results', 'image_features',
	'full_features', 'model-'+args.model, 'nsdsynthetic_colorstimuli_subj0'+
	str(args.subject))
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for i in tqdm(fmaps_list):
	features_nsdsynthetic.append(np.load(os.path.join(features_dir, i)))

# Format the image features to a numpy array
features_nsdsynthetic = np.asarray(features_nsdsynthetic)

# Z-score the image features
features_nsdsynthetic = scaler.transform(
	features_nsdsynthetic)

# Downsample the features with PCA
features_nsdsynthetic = pca.transform(features_nsdsynthetic)


# =============================================================================
# Save the PCA feature maps
# =============================================================================
image_features = {
	'features_nsdcore_train': features_nsdcore_train,
	'features_nsdcore_test': features_nsdcore_test,
	'features_nsdsynthetic': features_nsdsynthetic
	}

save_dir = os.path.join(args.project_dir, 'results', 'image_features',
	'pca_features', 'model-'+args.model)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'pca_features_sub-0' + str(args.subject) + '.npy'
np.save(os.path.join(save_dir, file_name), image_features)
