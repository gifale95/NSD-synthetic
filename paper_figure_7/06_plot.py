"""Plot the analyses results for individual NSD-core's ID/OOD and
NSD-synthetic's test splits.

Parameters
----------
subjects : list
	List of the used NSD subjects.
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
from copy import copy
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='vit_b_32', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Plot the MDS results
# =============================================================================
# Plot parameters
alpha = .75
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
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 1
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(204/255, 102/255, 0/255), (221/255, 204/255, 119/255),
	(51/255, 153/255, 153/255)]
gray = (75/255, 75/255, 75/255)
color_border = (127/255, 127/255, 127/255)

# Load the MDS results
dat_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'mds_all_subjects', 'zscore-'+str(args.zscore), 'mds.npy')
mds = np.load(dat_dir, allow_pickle=True).item()

# Create the figure
fig = plt.figure(figsize=(10, 10))

# Plot the fMRI responses for the NSD-core train images in MDS space
plt.scatter(mds['betas_mds_all_sub'][852:,0], mds['betas_mds_all_sub'][852:,1],
	s=50, color=gray, linewidths=0, alpha=.15)

# Plot the fMRI responses for the NSD-core ID test images in MDS space
plt.scatter(mds['betas_mds_all_sub'][:284,0], mds['betas_mds_all_sub'][:284,1],
	s=100, color=colors[0], linewidths=1, edgecolors=color_border, alpha=alpha,
	label='NSD-core ID test')

# Plot the fMRI responses for the NSD-core OOD test images in MDS space
plt.scatter(mds['betas_mds_all_sub'][284:568,0],
	mds['betas_mds_all_sub'][284:568,1], s=100, color=colors[1],
	edgecolors=color_border, linewidths=1, alpha=alpha,
	label='NSD-core OOD test')

# Plot the fMRI responses for the NSD-synthetic images in MDS space
plt.scatter(mds['betas_mds_all_sub'][568:852,0],
	mds['betas_mds_all_sub'][568:852,1], s=100, color=colors[2], linewidths=0.5,
	edgecolors=color_border, alpha=alpha, label='NSD-synthetic')

# x-axis
plt.xticks([])

# y-axis
plt.yticks([])

# Save the figure
file_name = 'mds.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()


# =============================================================================
# Plot the noise-ceiling-normalized explained variance on brain surfaces
# =============================================================================
# Plot parameters for colorbar
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage'

# Load the encoding accuracy results
data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'encoding_accuracy', 'zscore-'+str(args.zscore), 'model-'+args.model,
	'encoding_accuracy.npy')
results = np.load(data_dir, allow_pickle=True).item()

# Plot the NSD-core ID generalization
expl_var = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data = copy(results['lh_explained_variance_nsdcore_test_id'][s])
	rh_data = copy(results['rh_explained_variance_nsdcore_test_id'][s])
	# Remove vertices with noise ceiling values below a threshold, since they
	# cannot be interpreted in terms of modeling
	lh_idx_core_id = results['lh_nc_nsdcore_test_id'][s] > 0.3
	rh_idx_core_id = results['rh_nc_nsdcore_test_id'][s] > 0.3
	lh_idx_core_ood = results['lh_nc_nsdcore_test_ood'][s] > 0.3
	rh_idx_core_ood = results['rh_nc_nsdcore_test_ood'][s] > 0.3
	lh_idx_synt = results['lh_nc_nsdsynthetic'][s] > 0.3
	rh_idx_synt = results['rh_nc_nsdsynthetic'][s] > 0.3
	lh_idx = np.logical_and(lh_idx_core_id, lh_idx_core_ood)
	lh_idx = np.logical_and(lh_idx, lh_idx_synt)
	rh_idx = np.logical_and(rh_idx_core_id, rh_idx_core_ood)
	rh_idx = np.logical_and(rh_idx, rh_idx_synt)
	lh_data[~lh_idx] = np.nan
	rh_data[~rh_idx] = np.nan
	# Store the data
	expl_var.append(np.append(lh_data, rh_data))
# Plot the explained variance
vertex_data = cortex.Vertex(np.nanmean(expl_var, 0), subject, cmap='hot',
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
# Save the figure
file_name = 'explained_variance_nsdcore_id_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()
# Print the mean encoding accuracy
print('\n>>> Voxel average score NSD-core ID test: ' + str(np.nanmean(expl_var)))

# Plot the NSD-core OOD generalization
expl_var = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data = copy(results['lh_explained_variance_nsdcore_test_ood'][s])
	rh_data = copy(results['rh_explained_variance_nsdcore_test_ood'][s])
	# Remove vertices with noise ceiling values below a threshold, since they
	# cannot be interpreted in terms of modeling
	lh_idx_core_id = results['lh_nc_nsdcore_test_id'][s] > 0.3
	rh_idx_core_id = results['rh_nc_nsdcore_test_id'][s] > 0.3
	lh_idx_core_ood = results['lh_nc_nsdcore_test_ood'][s] > 0.3
	rh_idx_core_ood = results['rh_nc_nsdcore_test_ood'][s] > 0.3
	lh_idx_synt = results['lh_nc_nsdsynthetic'][s] > 0.3
	rh_idx_synt = results['rh_nc_nsdsynthetic'][s] > 0.3
	lh_idx = np.logical_and(lh_idx_core_id, lh_idx_core_ood)
	lh_idx = np.logical_and(lh_idx, lh_idx_synt)
	rh_idx = np.logical_and(rh_idx_core_id, rh_idx_core_ood)
	rh_idx = np.logical_and(rh_idx, rh_idx_synt)
	lh_data[~lh_idx] = np.nan
	rh_data[~rh_idx] = np.nan
	# Store the data
	expl_var.append(np.append(lh_data, rh_data))
# Plot the explained variance
vertex_data = cortex.Vertex(np.nanmean(expl_var, 0), subject, cmap='hot',
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
# Save the figure
file_name = 'explained_variance_nsdcore_ood_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()
# Print the mean encoding accuracy
print('\n>>> Voxel average score NSD-core OOD test: ' + str(np.nanmean(expl_var)))

# Plot the NSD-synthetic OOD generalization
expl_var = []
for s in range(len(args.subjects)):
	# Load the explained variance scores
	lh_data = copy(results['lh_explained_variance_nsdsynthetic'][s])
	rh_data = copy(results['rh_explained_variance_nsdsynthetic'][s])
	# Remove vertices with noise ceiling values below a threshold, since they
	# cannot be interpreted in terms of modeling
	lh_idx_core_id = results['lh_nc_nsdcore_test_id'][s] > 0.3
	rh_idx_core_id = results['rh_nc_nsdcore_test_id'][s] > 0.3
	lh_idx_core_ood = results['lh_nc_nsdcore_test_ood'][s] > 0.3
	rh_idx_core_ood = results['rh_nc_nsdcore_test_ood'][s] > 0.3
	lh_idx_synt = results['lh_nc_nsdsynthetic'][s] > 0.3
	rh_idx_synt = results['rh_nc_nsdsynthetic'][s] > 0.3
	lh_idx = np.logical_and(lh_idx_core_id, lh_idx_core_ood)
	lh_idx = np.logical_and(lh_idx, lh_idx_synt)
	rh_idx = np.logical_and(rh_idx_core_id, rh_idx_core_ood)
	rh_idx = np.logical_and(rh_idx, rh_idx_synt)
	lh_data[~lh_idx] = np.nan
	rh_data[~rh_idx] = np.nan
	# Store the data
	expl_var.append(np.append(lh_data, rh_data))
# Plot the explained variance
vertex_data = cortex.Vertex(np.nanmean(expl_var, 0), subject, cmap='hot',
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
# Save the figure
file_name = 'explained_variance_nsdsynthetic_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()
# Print the mean encoding accuracy
print('\n>>> Voxel average score NSD-synthetic: ' + str(np.nanmean(expl_var)))


# =============================================================================
# Plot the image identification accuracy
# =============================================================================
# Plot parameters
alpha_chance = .5
alpha_ci = 0.2
fontsize = 65
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

colors = [(204/255, 102/255, 0/255), (221/255, 204/255, 119/255),
	(51/255, 153/255, 153/255)]

# Load the identification results
data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'image_identification_accuracy', 'zscore-'+str(args.zscore), 'model-'+
	args.model, 'image_identification_accuracy.npy')
results = np.load(data_dir, allow_pickle=True).item()

# Create the figure
fig = plt.figure(figsize=(10, 15))

# Loop over conditions
conditions = [
	'ID generalization (NSD-core)',
	'OOD generalization (NSD-core)',
	'OOD generalization (NSD-synthetic)'
	]
scores_mean = []
for i in range(len(conditions)):

	# Add 1 so that the rankings start from 1 (now they start from 0)
	if i == 0:
		scores = results['rank_nsdcoreid'].flatten() + 1
	elif i == 1:
		scores = results['rank_nsdcoreood'].flatten() + 1
	elif i == 2:
		scores = results['rank_nsdsynthetic'].flatten() + 1

	# Plot the chance lines
	plt.plot([i-0.25, i+0.25], [142, 142], '--k', linewidth=2,
		alpha=alpha_chance, zorder=1)

	# Plot the image identification accuracies
	parts = plt.violinplot(
		dataset=scores,
		positions=[i],
		widths=0.75,
		showmeans=False,
		showextrema=False,
		points=len(scores)
		)

	# Set the violinplot colors
	for pc in parts['bodies']:
		pc.set_facecolor(colors[i])
		pc.set_edgecolor(None)
		pc.set_alpha(.5)

	# Plot the individual data points
	color_points = [(200/255, 200/255, 200/255)]
	color_edges = [(0/255, 0/255, 0/255)]
	x = np.random.normal(loc=i, scale=0.1, size=len(scores))
	plt.scatter(x, scores, s=100, color=colors[i], alpha=0.1,
		edgecolors=color_edges, linewidths=1, zorder=2)

	# Plot the mean
	x = i
	plt.scatter(x, np.mean(scores), s=500, color=colors[i], alpha=1,
		edgecolors='k', linewidths=2, zorder=2)
	scores_mean.append(int(np.round(np.mean(scores))))
	plt.text(x, np.mean(scores)+10, scores_mean[i], ha='center')

# y-axis
ticks = [1, 71, 142, 213, 284]
labels = [1, 71, 142, 213, 284]
plt.yticks(ticks=ticks, labels=labels)
plt.ylabel('Correct image rank', fontsize=fontsize)
plt.ylim(bottom=1, top=284)

# x-axis
ticks = np.arange(len(conditions))
labels = []
plt.xticks(ticks=ticks, labels=labels, fontsize=fontsize, rotation=0)
plt.xlabel('Test conditions', fontsize=fontsize)
#plt.setp(plt.gca().get_xticklabels(), ha='right') # Right align x-labels
plt.xlim(left=-0.6, right=2.6)

# Save the figure
file_name = 'identification_accuracy_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()
