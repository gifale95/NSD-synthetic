"""Downsample the image features with PCA.

Parameters
----------
subject : int
	Number of the used NSD subject.
data_ood_selection : str
	If 'fmri', the ID/OD splits are defined based on fMRI responses.
	If 'dnn', the ID/OD splits are defined based on DNN features.
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet', 'resnet50', 'moco', and 'vit_b_32'.
layer : str
	If 'all', apply PCA on the features from all model layers. If a layer name
	is given, PCA is applied on the features of that layer.
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
parser.add_argument('--data_ood_selection', default='fmri', type=str)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--layer', default='all', type=str)
parser.add_argument('--n_components', default=250, type=int)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Downsample image features with PCA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get the layer names
# =============================================================================
if args.model == 'alexnet':
	layers = [
		'features.2',
		'features.5',
		'features.7',
		'features.9',
		'features.12',
		'classifier.2',
		'classifier.5',
		'classifier.6'
		]

elif args.model == 'resnet50' or args.model == 'moco':
	layers = [
		'layer1.2.relu_2',
		'layer2.3.relu_2',
		'layer3.5.relu_2',
		'layer4.2.relu_2'
		]

elif args.model == 'vit_b_32':
	layers = [
		'encoder.layers.encoder_layer_0.add_1',
		'encoder.layers.encoder_layer_1.add_1',
		'encoder.layers.encoder_layer_2.add_1',
		'encoder.layers.encoder_layer_3.add_1',
		'encoder.layers.encoder_layer_4.add_1',
		'encoder.layers.encoder_layer_5.add_1',
		'encoder.layers.encoder_layer_6.add_1',
		'encoder.layers.encoder_layer_7.add_1',
		'encoder.layers.encoder_layer_8.add_1',
		'encoder.layers.encoder_layer_9.add_1',
		'encoder.layers.encoder_layer_10.add_1',
		'encoder.layers.encoder_layer_11.add_1'
		]


# =============================================================================
# Load the train/test splits
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'nsdcore_train_test_splits', 'data_ood_selection-'+args.data_ood_selection,
	'nsdcore_train_test_splits_subject-'+format(args.subject, '02')+'.npy')

train_test_splits = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# NSD-core train image features
# =============================================================================
features_dir = os.path.join(args.project_dir, 'results', 'image_features',
	'full_features', 'model-'+args.model, 'nsdcore')

features_nsdcore_train = []
train_img_num = train_test_splits['train_img_num']
for i in tqdm(train_img_num):
	features = np.load(os.path.join(features_dir, 'img_'+format(i,'06')+'.npy'),
		allow_pickle=True).item()
	if args.layer == 'all':
		ft = np.empty(0, dtype=np.float32)
		for layer in layers:
			ft = np.append(ft, np.reshape(features[layer], -1))
		features_nsdcore_train.append(ft)
		del ft
	else:
		features_nsdcore_train.append(np.reshape(features[args.layer], -1))
	del features
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
# NSD-core ID test image features
# =============================================================================
features_nsdcore_test_id = []
test_img_num_id = train_test_splits['test_img_num_id']
for i in tqdm(test_img_num_id):
	features = np.load(os.path.join(features_dir, 'img_'+format(i,'06')+'.npy'),
		allow_pickle=True).item()
	if args.layer == 'all':
		ft = np.empty(0, dtype=np.float32)
		for layer in layers:
			ft = np.append(ft, np.reshape(features[layer], -1))
		features_nsdcore_test_id.append(ft)
		del ft
	else:
		features_nsdcore_test_id.append(np.reshape(features[args.layer], -1))
	del features
features_nsdcore_test_id = np.asarray(features_nsdcore_test_id)

# Z-score the image features
features_nsdcore_test_id = scaler.transform(features_nsdcore_test_id)

# Downsample the features with PCA
features_nsdcore_test_id = pca.transform(features_nsdcore_test_id)


# =============================================================================
# NSD-core OOD test image features
# =============================================================================
features_nsdcore_test_ood = []
test_img_num_ood = train_test_splits['test_img_num_ood']
for i in tqdm(test_img_num_ood):
	features = np.load(os.path.join(features_dir, 'img_'+format(i,'06')+'.npy'),
		allow_pickle=True).item()
	if args.layer == 'all':
		ft = np.empty(0, dtype=np.float32)
		for layer in layers:
			ft = np.append(ft, np.reshape(features[layer], -1))
		features_nsdcore_test_ood.append(ft)
		del ft
	else:
		features_nsdcore_test_ood.append(np.reshape(features[args.layer], -1))
	del features
features_nsdcore_test_ood = np.asarray(features_nsdcore_test_ood)

# Z-score the image features
features_nsdcore_test_ood = scaler.transform(features_nsdcore_test_ood)

# Downsample the features with PCA
features_nsdcore_test_ood = pca.transform(features_nsdcore_test_ood)


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
	features = np.load(os.path.join(features_dir, i), allow_pickle=True).item()
	if args.layer == 'all':
		ft = np.empty(0, dtype=np.float32)
		for layer in layers:
			ft = np.append(ft, np.reshape(features[layer], -1))
		features_nsdsynthetic.append(ft)
		del ft
	else:
		features_nsdsynthetic.append(np.reshape(features[args.layer], -1))
	del features

# Load the NSD-synthetic colorstimuli features
fmaps_dir = os.path.join(args.project_dir, 'results', 'image_features',
	'full_features', 'model-'+args.model, 'nsdsynthetic_colorstimuli_subj0'+
	str(args.subject))
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for i in tqdm(fmaps_list):
	features = np.load(os.path.join(features_dir, i), allow_pickle=True).item()
	if args.layer == 'all':
		ft = np.empty(0, dtype=np.float32)
		for layer in layers:
			ft = np.append(ft, np.reshape(features[layer], -1))
		features_nsdsynthetic.append(ft)
		del ft
	else:
		features_nsdsynthetic.append(np.reshape(features[args.layer], -1))
	del features

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
	'features_nsdcore_test_id': features_nsdcore_test_id,
	'features_nsdcore_test_ood': features_nsdcore_test_ood,
	'features_nsdsynthetic': features_nsdsynthetic
	}

save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'pca_features', 'data_ood_selection-'+args.data_ood_selection, 'model-'+
	args.model, 'layer-'+args.layer)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'pca_features_sub-0' + str(args.subject) + '.npy'
np.save(os.path.join(save_dir, file_name), image_features)
