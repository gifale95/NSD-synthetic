"""Plot the encoding models' in-distribution (NSD-core) and out-of-distribution
(NSD-synthetic) encoding accuracy.

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
import numpy as np
from copy import copy
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--train_test_session_control', type=int, default=0)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--layer', default='all', type=str)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding accuracy results
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results',
	'train_test_session_control-'+str(args.train_test_session_control),
	'encoding_accuracy', 'zscore-'+str(args.zscore), 'model-'+args.model,
	'layer-'+args.layer, 'regression-'+args.regression, 'encoding_accuracy.npy')
results = np.load(data_dir, allow_pickle=True).item()


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
	'_model-'+args.model + '_layer-'+args.layer + '_regression-' + \
	args.regression


# =============================================================================
# Plot the in-distribution (NSD-core) encoding accuracy
# =============================================================================
# r2
r2_core = np.append(
	np.mean(results['lh_r2_nsdcore_test'], 0),
	np.mean(results['rh_r2_nsdcore_test'], 0))
vertex_data = cortex.Vertex(r2_core, subject, cmap='hot', vmin=0, vmax=.8,
	with_colorbar=True)
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
file_name = 'nsdcore_r2_' + param + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()

# Noise ceiling
nc_core = np.append(
	np.nanmean(results['lh_nc_nsdcore_test'], 0),
	np.nanmean(results['rh_nc_nsdcore_test'], 0))
vertex_data = cortex.Vertex(nc_core, subject, cmap='hot', vmin=0, vmax=.8,
	with_colorbar=True)
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
file_name = 'nsdcore_' + param + '_noise_ceiling.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()

# Explained variance
expl_var_core = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data = copy(results['lh_explained_variance_nsdcore_test'][s])
	rh_data = copy(results['rh_explained_variance_nsdcore_test'][s])
	# Remove vertices with noise ceiling values below a threshold, since they
	# cannot be interpreted in terms of modeling
	lh_idx = results['lh_nc_nsdcore_test'][s] < 0.3
	rh_idx = results['rh_nc_nsdcore_test'][s] < 0.3
	lh_data[lh_idx] = np.nan
	rh_data[rh_idx] = np.nan
	# Store the data
	expl_var_core.append(np.append(lh_data, rh_data))
# Plot the explained variance
vertex_data = cortex.Vertex(np.nanmean(expl_var_core, 0), subject, cmap='hot',
	vmin=0, vmax=100, with_colorbar=True)
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
file_name = 'nsdcore_explained_variance_' + param + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()
# Print the mean encoding accuracy
print(np.nanmean(expl_var_core))


# =============================================================================
# Plot the out-of-distribution (NSD-synthetic) encoding accuracy
# =============================================================================
# r2
r2_synt = np.append(
	np.mean(results['lh_r2_nsdsynthetic'], 0),
	np.mean(results['rh_r2_nsdsynthetic'], 0))
vertex_data = cortex.Vertex(r2_synt, subject, cmap='hot', vmin=0, vmax=.8,
	with_colorbar=True)
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
file_name = 'nsdsynthetic_r2_' + param + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()

# Noise ceiling
nc_synt = np.append(
	np.mean(results['lh_nc_nsdsynthetic'], 0),
	np.mean(results['rh_nc_nsdsynthetic'], 0))
vertex_data = cortex.Vertex(nc_synt, subject, cmap='hot', vmin=0, vmax=.8,
	with_colorbar=True)
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
file_name = 'nsdsynthetic_' + param + '_noise_ceiling.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()

# Explained variance
expl_var_synt = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data = copy(results['lh_explained_variance_nsdsynthetic'][s])
	rh_data = copy(results['rh_explained_variance_nsdsynthetic'][s])
	# Remove vertices with noise ceiling values below a threshold, since they
	# cannot be interpreted in terms of modeling
	lh_idx = results['lh_nc_nsdsynthetic'][s] < 0.3
	rh_idx = results['rh_nc_nsdsynthetic'][s] < 0.3
	lh_data[lh_idx] = np.nan
	rh_data[rh_idx] = np.nan
	# Store the data
	expl_var_synt.append(np.append(lh_data, rh_data))
# Plot the explained variance
vertex_data = cortex.Vertex(np.nanmean(expl_var_synt, 0), subject, cmap='hot',
	vmin=0, vmax=100, with_colorbar=True)
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
file_name = 'nsdsynthetic_explained_variance_' + param + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()
# Print the mean encoding accuracy
print(np.nanmean(expl_var_synt))


# =============================================================================
# NSD-core minus NSD-synthetic
# =============================================================================
# Compute the difference between NSD-core and NSD-synthetic's explained
# variances
delta_expl_var = []
for s in range(len(args.subjects)):
	# Compute the difference between NSD-core and NSD-synthetic
	lh_data = results['lh_explained_variance_nsdcore_test'][s] - \
		results['lh_explained_variance_nsdsynthetic'][s]
	rh_data = results['rh_explained_variance_nsdcore_test'][s] - \
		results['rh_explained_variance_nsdsynthetic'][s]
	# Remove vertices with noise ceiling values below a threshold across both
	# NSD-synthetic and NSD-core, since they cannot be interpreted in terms of
	# modeling
	lh_idx_core = results['lh_nc_nsdcore_test'][s] > 0.3
	rh_idx_core = results['rh_nc_nsdcore_test'][s] > 0.3
	lh_idx_synt = results['lh_nc_nsdsynthetic'][s] > 0.3
	rh_idx_synt = results['rh_nc_nsdsynthetic'][s] > 0.3
	lh_idx = np.logical_and(lh_idx_core, lh_idx_synt)
	rh_idx = np.logical_and(rh_idx_core, rh_idx_synt)
	lh_data[~lh_idx] = np.nan
	rh_data[~rh_idx] = np.nan
	# Store the data
	delta_expl_var.append(np.append(lh_data, rh_data))
	del lh_data, rh_data
delta_expl_var = np.asarray(delta_expl_var)

# Plot
vertex_data = cortex.Vertex(np.nanmean(delta_expl_var, 0),
	subject, cmap='RdBu_r', vmin=-70, vmax=70, with_colorbar=True)
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
file_name = 'nsdcore_minus_nsdsynthetic_explained_variance_' + param + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()

# Compute the p-value of the difference
print(np.nanmean(delta_expl_var))
delta_expl_var_mean = np.nanmean(delta_expl_var, 1)
pval = ttest_1samp(delta_expl_var_mean, 0, alternative='greater')[1]


# =============================================================================
# Scatterplot of vertex-wise r² scores against the noise ceiling
# =============================================================================
# Load the visual stream ROI indices
data_dir = os.path.join(args.project_dir, 'results',
	'train_test_session_control-'+str(args.train_test_session_control),
	'fmri_betas', 'zscore-'+str(args.zscore), 'sub-01', 'meatadata_nsdcore.npy')
metadata = np.load(data_dir, allow_pickle=True).item()
# Left hemisphere
lh_streams = {}
lh_streams['early'] = metadata['lh_fsaverage_rois']['early']
intermediate = []
intermediate.append(metadata['lh_fsaverage_rois']['midventral'])
intermediate.append(metadata['lh_fsaverage_rois']['midlateral'])
intermediate.append(metadata['lh_fsaverage_rois']['midparietal'])
intermediate = np.concatenate(intermediate)
intermediate.sort()
lh_streams['intermediate'] = intermediate
lh_streams['ventral'] = metadata['lh_fsaverage_rois']['ventral']
lh_streams['lateral'] = metadata['lh_fsaverage_rois']['lateral']
lh_streams['parietal'] = metadata['lh_fsaverage_rois']['parietal']
# Right hemisphere
rh_streams = {}
rh_streams['early'] = metadata['rh_fsaverage_rois']['early']
intermediate = []
intermediate.append(metadata['rh_fsaverage_rois']['midventral'])
intermediate.append(metadata['rh_fsaverage_rois']['midlateral'])
intermediate.append(metadata['rh_fsaverage_rois']['midparietal'])
intermediate = np.concatenate(intermediate)
intermediate.sort()
rh_streams['intermediate'] = intermediate
rh_streams['ventral'] = metadata['rh_fsaverage_rois']['ventral']
rh_streams['lateral'] = metadata['rh_fsaverage_rois']['lateral']
rh_streams['parietal'] = metadata['rh_fsaverage_rois']['parietal']

# Get the results
r2_synthetic = []
r2_core = []
nc_synthetic = []
nc_core = []
roi = []
colors = []
for s in range(len(args.subjects)):
	# Only retain results for vertices with noise ceiling above threshold in
	# both NSD-synthetic and NSD-core
	lh_idx_core = results['lh_nc_nsdcore_test'][s] > 0.3
	rh_idx_core = results['rh_nc_nsdcore_test'][s] > 0.3
	lh_idx_synt = results['lh_nc_nsdsynthetic'][s] > 0.3
	rh_idx_synt = results['rh_nc_nsdsynthetic'][s] > 0.3
	lh_idx = np.logical_and(lh_idx_core, lh_idx_synt)
	rh_idx = np.logical_and(rh_idx_core, rh_idx_synt)
	# Store the results
	r2_synthetic.append(results['lh_r2_nsdsynthetic'][s][lh_idx])
	r2_synthetic.append(results['rh_r2_nsdsynthetic'][s][rh_idx])
	r2_core.append(results['lh_r2_nsdcore_test'][s][lh_idx])
	r2_core.append(results['rh_r2_nsdcore_test'][s][rh_idx])
	nc_synthetic.append(results['lh_nc_nsdsynthetic'][s][lh_idx])
	nc_synthetic.append(results['rh_nc_nsdsynthetic'][s][rh_idx])
	nc_core.append(results['lh_nc_nsdcore_test'][s][lh_idx])
	nc_core.append(results['rh_nc_nsdcore_test'][s][rh_idx])
	# Assign each vertex to its ROI
	idx = np.append(np.where(lh_idx)[0], np.where(rh_idx)[0])
	for i in idx:
		if i in lh_streams['early']:
			roi.append('early')
			colors.append((67/255, 147/255, 195/255))
		elif i in lh_streams['intermediate']:
			roi.append('intermediate')
			colors.append((209/255, 230/255, 241/255))
		elif i in lh_streams['ventral']:
			roi.append('ventral')
			colors.append((253/255, 219/255, 199/255))
		elif i in lh_streams['lateral']:
			roi.append('lateral')
			colors.append((214/255, 96/255, 77/255))
		elif i in lh_streams['parietal']:
			roi.append('parietal')
			colors.append((103/255, 0/255, 31/255))
		else:
			roi.append('none')
			colors.append((75/255, 75/255, 75/255))
# Concatenate the results across subejcts
r2_synthetic = np.concatenate(r2_synthetic)
r2_core = np.concatenate(r2_core)
nc_synthetic = np.concatenate(nc_synthetic)
nc_core = np.concatenate(nc_core)

# Plot parameters
fontsize = 50
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['lines.markersize'] = 2
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 1
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'

# Plot the r² scores against the noise ceiling scores
fig, axs = plt.subplots(figsize=(21, 13), nrows=1, ncols=2, sharex=True,
	sharey=True)
axs = np.reshape(axs, (-1))
for i in range(len(axs)):
	# Plot the results
	if i == 0:
		# Vertex-wise results
		axs[i].scatter(nc_core, r2_core, s=1, color=colors, alpha=1)
		# Vertex-median results
		axs[i].scatter(np.median(nc_core), np.median(r2_core), s=300,
			marker='X', color='k', alpha=1, zorder=2)
	elif i == 1:
		# Vertex-wise results
		axs[i].scatter(nc_synthetic, r2_synthetic, s=1, color=colors, alpha=1)
		# Vertex-median results
		axs[i].scatter(np.median(nc_synthetic), np.median(r2_synthetic), s=300,
			marker='X', color='k', alpha=1, zorder=2)
	axs[i].set_aspect('equal')
	# Plot diagonal dashed line
	axs[i].plot(np.arange(-1,1.1,.1), np.arange(-1,1.1,.1), '--k', linewidth=2,
		alpha=.5, label='_nolegend_', zorder=1)
	# y-axis
	if i in [0]:
		ticks = [0.2, 0.4, 0.6, 0.8, 1]
		labels = ['.2', '.4', '.6', '.8', '1']
		axs[i].set_ylabel('Explained variance ($r²$)', fontsize=fontsize)
		plt.yticks(ticks=ticks, labels=labels)
	axs[i].set_ylim(bottom=0, top=1)
	# x-axis
	if i in [0, 1]:
		ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
		labels = ['0', '.2', '.4', '.6', '.8', '1']
		axs[i].set_xlabel('Noise ceiling ($r²$)', fontsize=fontsize)
		plt.xticks(ticks=ticks, labels=labels, fontsize=fontsize)
	axs[i].set_xlim(left=0, right=1)

# Save the figure
file_name = 'encoding_accuracy_scatterplots_' + param + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
file_name = 'encoding_accuracy_scatterplots_' + param + '.png'
fig.savefig(file_name, dpi=600, bbox_inches='tight', transparent=True,
	format='png')
plt.close()
