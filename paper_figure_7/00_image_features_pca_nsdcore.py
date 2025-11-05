"""Downsample the image features with PCA, for all NSD-core images seen by a
subject.

Parameters
----------
subject : int
	Number of the used NSD subject.
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet', 'resnet50', 'moco', and 'vit_b_32'.
layer : str
	If 'all', apply PCA on the features from all model layers. If a layer name
	is given, PCA is applied on the features of that layer.
n_components : int
	Number of PCA components retained.
nsd_dir : str
	Directory of the NSD.
project_dir : str
	Directory of the project folder.

"""

import argparse
import numpy as np
import os
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--model', default='vit_b_32', type=str)
parser.add_argument('--layer', default='all', type=str)
parser.add_argument('--n_components', default=250, type=int)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()

print('>>> Downsample image features with PCA <<<')
print('\nInput arguments:')
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
# NSD-core train image features
# =============================================================================
features_dir = os.path.join(args.project_dir, 'results', 'image_features',
	'full_features', 'model-'+args.model, 'nsdcore')

features_nsdcore = []
for i in tqdm(img_num):
	features = np.load(os.path.join(features_dir, 'img_'+format(i,'06')+'.npy'),
		allow_pickle=True).item()
	if args.layer == 'all':
		ft = np.empty(0, dtype=np.float32)
		for layer in layers:
			ft = np.append(ft, np.reshape(features[layer], -1))
		features_nsdcore.append(ft)
		del ft
	else:
		features_nsdcore.append(np.reshape(features[args.layer], -1))
	del features
features_nsdcore = np.asarray(features_nsdcore)

# Z-score the image features
scaler = StandardScaler()
scaler.fit(features_nsdcore)
features_nsdcore = scaler.transform(features_nsdcore)

# Downsample the features with PCA
pca = PCA(n_components=args.n_components, random_state=20200220)
pca.fit(features_nsdcore)
features_nsdcore = pca.transform(features_nsdcore)


# =============================================================================
# Save the PCA feature maps
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'image_features', 'pca_features', 'model-'+args.model, 'layer-'+args.layer)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'pca_features_sub-0' + str(args.subject) + '.npy'
np.save(os.path.join(save_dir, file_name), features_nsdcore)
