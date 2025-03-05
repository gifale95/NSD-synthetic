"""Extract and save the NSD-synthetic image features, using deep neural network
models.

Parameters
----------
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet', 'resnet50', 'moco', and 'vit_b_32'.
nsd_dir : str
	Directory of the NSD.
project_dir : str
	Directory of the project folder.

"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor
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

print('>>> Extract image features NSD-synthetic <<<')
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

# ResNet-50
elif args.model == 'resnet50':
	# Load the model
	model = torchvision.models.resnet50(weights='DEFAULT')
	# Select the used layers for feature extraction
	#nodes, _ = get_graph_node_names(model)
	model_layers = [
		'layer1.2.relu_2',
		'layer2.3.relu_2',
		'layer3.5.relu_2',
		'layer4.2.relu_2',
		'fc'
		]

# MoCo
elif args.model == 'moco':
	# Load the ResNet-50 model
	model = torchvision.models.resnet50(weights='DEFAULT')
	# Load the MoCo weights (https://github.com/facebookresearch/moco)
	checkpoint = torch.load('moco_v2_800ep_pretrain.pth.tar',
		map_location='cpu')['state_dict']
	# Remove "module.encoder_q." prefix if present
	state_dict = {k.replace('module.encoder_q.', ''): v for k, v in checkpoint.items()}
	# Extract FC weights
	fc_state_dict = {k: v for k, v in state_dict.items() if k.startswith("fc.")}
	# Define MoCo's FC structure
	if 'fc.0.weight' in fc_state_dict:  # Multi-layer FC (MoCo v2)
		dims = [fc_state_dict['fc.0.weight'].shape[1],
			fc_state_dict['fc.0.weight'].shape[0],
			fc_state_dict['fc.2.weight'].shape[0]]
		model.fc = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(),
			nn.Linear(dims[1], dims[2]))
	else:  # Single-layer FC (MoCo v1)
		dims = [fc_state_dict['fc.weight'].shape[1],
			fc_state_dict['fc.weight'].shape[0]]
		model.fc = nn.Linear(*dims[::-1])
	# Load weights (excluding FC first)
	model.load_state_dict({k: v for k, v in state_dict.items() if k not in fc_state_dict}, strict=False)
	model.fc.load_state_dict({k.replace('fc.', ''): v for k, v in fc_state_dict.items()}, strict=True)
	# Select the used layers for feature extraction
	#nodes, _ = get_graph_node_names(model)
	model_layers = [
		'layer1.2.relu_2',
		'layer2.3.relu_2',
		'layer3.5.relu_2',
		'layer4.2.relu_2',
		'fc.2'
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
# Extract the feature maps of the nsdsynthetic stimuli
# =============================================================================
stimuli_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
	'nsdsynthetic')
stimuli_files = os.listdir(stimuli_dir)
stimuli_files.sort()

for s in tqdm(stimuli_files):
	sf = h5py.File(os.path.join(stimuli_dir, s), 'r')
	sdataset = sf.get('imgBrick')
	save_dir = os.path.join(args.project_dir, 'results', 'image_features',
		'full_features', 'model-'+args.model, s[:-5])
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	with torch.no_grad():
		for i, img in enumerate(tqdm(sdataset, leave=False)):
			# Preprocess the images
			img = (np.sqrt(img/255)*255).astype(np.uint8)
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
