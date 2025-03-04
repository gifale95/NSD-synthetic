"""Extract and save the NSD-core image features, using deep neural network
models.

Parameters
----------
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet' and 'vit_b_32'.
nsd_dir : str
	Directory of the NSD.
project_dir : str
	Directory of the project folder.

"""

import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import os
import h5py
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()

print('>>> Extract image features NSD-core <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Define the image preprocessing
# =============================================================================
transform = trn.Compose([
	trn.Lambda(lambda img: trn.CenterCrop(min(img.size))(img)),
	trn.Resize((224,224)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Load the deep neural network model
# =============================================================================
# AlexNet
if args.model == 'alexnet':
	# Load the model
	model = torchvision.models.alexnet(weights='DEFAULT')
	# Select the used layers for feature extraction
	#nodes, _ = get_graph_node_names(model)
	model_layers = [
		'features.2',
		'features.5',
		'features.7',
		'features.9',
		'features.12',
		'classifier.2',
		'classifier.5',
		'classifier.6'
		]

# vit_b_32
elif args.model == 'vit_b_32':
	# Load the model
	model = torchvision.models.vit_b_32(weights='DEFAULT')
	# Select the used layers for feature extraction
	#nodes, _ = get_graph_node_names(model)
	model_layers = [
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

# Create the feature extractor
feature_extractor = create_feature_extractor(model, return_nodes=model_layers)
feature_extractor.to(device)
feature_extractor.eval()


# =============================================================================
# Access the NSD-core images
# =============================================================================
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
	'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')


# =============================================================================
# Extract and store the image features
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'image_features',
	'full_features', 'model-'+args.model, 'nsdcore')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

with torch.no_grad():
	for i, img in enumerate(tqdm(sdataset, leave=False)):
		# Preprocess the images
		img = Image.fromarray(img).convert('RGB')
		img = transform(img).unsqueeze(0)
		img = img.to(device)
		# Extract the features
		ft = feature_extractor(img)
		# Flatten the features
		ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
		file_name = 'img_' + format(i, '06') + '.npy'
		np.save(os.path.join(save_dir, file_name),
			np.squeeze(ft.cpu().detach().numpy()))
		del ft
