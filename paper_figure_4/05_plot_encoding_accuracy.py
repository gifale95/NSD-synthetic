"""Plot the encoding models' in-distribution (NSD-core) and out-of-distribution
(NSD-synthetic) encoding accuracy.

Parameters
----------
subjects : list
	List of the used NSD subjects.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
model : str
	Name of deep neural network model used to extract the image features.
	Available options are 'alexnet' and 'vit_b_32'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from copy import copy
import cortex
import cortex.polyutils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding accuracy results
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'encoding_accuracy',
	'zscored-'+str(args.zscore), 'model-'+args.model, 'encoding_accuracy.npy')
results = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Plot parameters for colorbar
# =============================================================================
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
subject = 'fsaverage'


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
file_name = 'nsdcore_r2_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')

# Noise ceiling
nc_core = np.append(
	np.nanmean(results['lh_nc_nsdcore_test_284'], 0),
	np.nanmean(results['rh_nc_nsdcore_test_284'], 0))
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
file_name = 'nsdcore_noise_ceiling_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')

# Explained variance
expl_var_core = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data = copy(results['lh_explained_variance_nsdcore_test_284'][s])
	rh_data = copy(results['rh_explained_variance_nsdcore_test_284'][s])
	# Remove vertices with noise ceiling values below a threshold, since they
	# cannot be interpreted in terms of modeling
	lh_idx = results['lh_nc_nsdcore_test_284'][s] < 0.2
	rh_idx = results['rh_nc_nsdcore_test_284'][s] < 0.2
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
file_name = 'nsdcore_explained_variance_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')


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
file_name = 'nsdsynthetic_r2_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')

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
file_name = 'nsdsynthetic_noise_ceiling_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')

# Explained variance
expl_var_synt = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data = copy(results['lh_explained_variance_nsdsynthetic'][s])
	rh_data = copy(results['rh_explained_variance_nsdsynthetic'][s])
	# Remove vertices with noise ceiling values below a threshold, since they
	# cannot be interpreted in terms of modeling
	lh_idx = results['lh_nc_nsdsynthetic'][s] < 0.2
	rh_idx = results['rh_nc_nsdsynthetic'][s] < 0.2
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
file_name = 'nsdsynthetic_explained_variance_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')


# =============================================================================
# NSD-core minus NSD-synthetic
# =============================================================================
# Compute the difference between NSD-core and NSD-synthetic's explained
# variances
delta_expl_var = []
for s in range(len(args.subjects)):
	# Compute the difference between NSD-core and NSD-synthetic
	lh_data = results['lh_explained_variance_nsdcore_test_284'][s] - \
		results['lh_explained_variance_nsdsynthetic'][s]
	rh_data = results['rh_explained_variance_nsdcore_test_284'][s] - \
		results['rh_explained_variance_nsdsynthetic'][s]
	# Remove vertices with noise ceiling values below a threshold across both
	# NSD-synthetic and NSD-core, since they cannot be interpreted in terms of
	# modeling
	lh_idx_core = results['lh_nc_nsdcore_test_284'][s] > 0.2
	rh_idx_core = results['rh_nc_nsdcore_test_284'][s] > 0.2
	lh_idx_synt = results['lh_nc_nsdsynthetic'][s] > 0.2
	rh_idx_synt = results['rh_nc_nsdsynthetic'][s] > 0.2
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
file_name = 'nsdcore_minus_nsdsynthetic_explained_variance_model-' + \
	args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
