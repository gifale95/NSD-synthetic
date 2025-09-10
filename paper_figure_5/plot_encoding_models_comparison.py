"""Compare the ID and OOD generalization performances of different encoding
models.

Parameters
----------
subjects : list
	List of the used NSD subjects.
train_test_session_control : int
	If '1', use the train and test splits consist of image conditions from
	non-overlapping fMRI scan sessions.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
models : list
	List of the of deep neural network models used to extract the image
	features, and that will be compared. Available options are 'alexnet',
	'vit_b_32', 'resnet50' and 'moco'.
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
import numpy as np
from copy import copy
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--train_test_session_control', type=int, default=0)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--models', default=['vit_b_32', 'alexnet'], type=list)
parser.add_argument('--layer', default='all', type=str)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Load the ID & OOD encoding accuracy results
# =============================================================================
results = {}

for m in args.models:

	data_dir = os.path.join(args.project_dir, 'results',
		'train_test_session_control-'+str(args.train_test_session_control),
		'encoding_accuracy', 'zscore-'+str(args.zscore), 'model-'+m, 'layer-'+
		args.layer, 'regression-'+args.regression, 'encoding_accuracy.npy')
	results[m] = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Plot parameters
# =============================================================================
# Plot parameters for colorbar
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage'

# Plot file names
param = 'train_test_session_control-' + str(args.train_test_session_control) + \
	'_layer-' + args.layer + '_regression-' + args.regression


# =============================================================================
# ID generalization (NSD-core)
# =============================================================================
key = 'resnet50'
val = results[key]

key = 'moco'
val = results[key]


# Explained variance
for key, val in results.items():
	expl_var_core = []
	for s in range(len(args.subjects)):
		# Load the explained variance scores
		lh_data = copy(val['lh_explained_variance_nsdcore_test'][s])
		rh_data = copy(val['rh_explained_variance_nsdcore_test'][s])
		# Remove vertices with noise ceiling values below a certain threshold,
		# since they cannot be interpreted in terms of modeling
		lh_idx = val['lh_nc_nsdcore_test'][s] < 0.3
		rh_idx = val['rh_nc_nsdcore_test'][s] < 0.3
		lh_data[lh_idx] = np.nan
		rh_data[rh_idx] = np.nan
		# Store the data
		expl_var_core.append(np.append(lh_data, rh_data))
	# Plot the explained variance
	vertex_data = cortex.Vertex(np.nanmean(expl_var_core, 0), subject,
		cmap='hot', vmin=0, vmax=100, with_colorbar=True)
	fig = cortex.quickshow(vertex_data,
	#	height=500, # Increase resolution of map and ROI contours
		with_curvature=True,
		curvature_brightness=0.5,
		with_rois=True,
		with_labels=False,
		linewidth=5,
		linecolor=(1, 1, 1),
		with_colorbar=True
		)
	plt.show()
	file_name = 'nsdcore_explained_variance_' + param + '_model-' + key + \
		'.svg'
	fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
		format='svg')
	plt.close()

# Delta explained variance
lh_delta_expl_var_core = []
rh_delta_expl_var_core = []
delta_expl_var_core = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data_model_1 = copy(
		results[args.models[0]]['lh_explained_variance_nsdcore_test'][s])
	lh_data_model_2 = copy(
		results[args.models[1]]['lh_explained_variance_nsdcore_test'][s])
	lh_data = lh_data_model_1 - lh_data_model_2
	rh_data_model_1 = copy(
		results[args.models[0]]['rh_explained_variance_nsdcore_test'][s])
	rh_data_model_2 = copy(
		results[args.models[1]]['rh_explained_variance_nsdcore_test'][s])
	rh_data = rh_data_model_1 - rh_data_model_2
	# Remove vertices with noise ceiling values below a certain threshold,
	# since they cannot be interpreted in terms of modeling
	lh_idx = results[args.models[0]]['lh_nc_nsdcore_test'][s] < 0.3
	rh_idx = results[args.models[0]]['rh_nc_nsdcore_test'][s] < 0.3
	lh_data[lh_idx] = np.nan
	rh_data[rh_idx] = np.nan
	# Store the data
	lh_delta_expl_var_core.append(lh_data)
	rh_delta_expl_var_core.append(rh_data)
	delta_expl_var_core.append(np.append(lh_data, rh_data))
# Plot the explained variance
vertex_data = cortex.Vertex(np.nanmean(delta_expl_var_core, 0), subject,
	cmap='RdBu_r', vmin=-25, vmax=25, with_colorbar=True)
fig = cortex.quickshow(vertex_data,
#	height=500, # Increase resolution of map and ROI contours
	with_curvature=True,
	curvature_brightness=0.5,
	with_rois=True,
	with_labels=False,
	linewidth=5,
	linecolor=(1, 1, 1),
	with_colorbar=True
	)
plt.show()
file_name = 'nsdcore_delta_explained_variance_' + param + '_' + \
	args.models[0] + '-' + args.models[1] + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()


# =============================================================================
# OOD generalization (NSD-synthetic)
# =============================================================================
# Explained variance
for key, val in results.items():
	expl_var_synt = []
	for s in range(len(args.subjects)):
		# Load the explained variance scores
		lh_data = copy(val['lh_explained_variance_nsdsynthetic'][s])
		rh_data = copy(val['rh_explained_variance_nsdsynthetic'][s])
		# Remove vertices with noise ceiling values below a certain threshold,
		# since they cannot be interpreted in terms of modeling
		lh_idx = val['lh_nc_nsdsynthetic'][s] < 0.3
		rh_idx = val['rh_nc_nsdsynthetic'][s] < 0.3
		lh_data[lh_idx] = np.nan
		rh_data[rh_idx] = np.nan
		# Store the data
		expl_var_synt.append(np.append(lh_data, rh_data))
	# Plot the explained variance
	vertex_data = cortex.Vertex(np.nanmean(expl_var_synt, 0), subject,
		cmap='hot', vmin=0, vmax=100, with_colorbar=True)
	fig = cortex.quickshow(vertex_data,
	#	height=500, # Increase resolution of map and ROI contours
		with_curvature=True,
		curvature_brightness=0.5,
		with_rois=True,
		with_labels=False,
		linewidth=5,
		linecolor=(1, 1, 1),
		with_colorbar=True
		)
	plt.show()
	file_name = 'nsdsynthetic_explained_variance_' + param + '_model-' + \
		key + '.svg'
	fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
		format='svg')
	plt.close()

# Delta explained variance
lh_delta_expl_var_synthetic = []
rh_delta_expl_var_synthetic = []
delta_expl_var_synthetic = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data_model_1 = copy(
		results[args.models[0]]['lh_explained_variance_nsdsynthetic'][s])
	lh_data_model_2 = copy(
		results[args.models[1]]['lh_explained_variance_nsdsynthetic'][s])
	lh_data = lh_data_model_1 - lh_data_model_2
	rh_data_model_1 = copy(
		results[args.models[0]]['rh_explained_variance_nsdsynthetic'][s])
	rh_data_model_2 = copy(
		results[args.models[1]]['rh_explained_variance_nsdsynthetic'][s])
	rh_data = rh_data_model_1 - rh_data_model_2
	# Remove vertices with noise ceiling values below a certain threshold,
	# since they cannot be interpreted in terms of modeling
	lh_idx = results[args.models[0]]['lh_nc_nsdsynthetic'][s] < 0.3
	rh_idx = results[args.models[0]]['rh_nc_nsdsynthetic'][s] < 0.3
	lh_data[lh_idx] = np.nan
	rh_data[rh_idx] = np.nan
	# Store the data
	lh_delta_expl_var_synthetic.append(lh_data)
	rh_delta_expl_var_synthetic.append(rh_data)
	delta_expl_var_synthetic.append(np.append(lh_data, rh_data))
# Plot the explained variance
vertex_data = cortex.Vertex(np.nanmean(delta_expl_var_synthetic, 0), subject,
	cmap='RdBu_r', vmin=-25, vmax=25, with_colorbar=True)
fig = cortex.quickshow(vertex_data,
#	height=500, # Increase resolution of map and ROI contours
	with_curvature=True,
	curvature_brightness=0.5,
	with_rois=True,
	with_labels=False,
	linewidth=5,
	linecolor=(1, 1, 1),
	with_colorbar=True
	)
plt.show()
file_name = 'nsdynthetic_delta_explained_variance_' + param + '_' + \
	args.models[0] + '-' + args.models[1] + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()
