"""Plot the MDS univariate and multivariate NSD-synthetic fMRI responses
analsyes.

Parameters
----------
subjects : list
	List with all used NSD subject.
rois : list
	List with all used NSD ROIs.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept vertices: only vertices above this
	threshold are used.
nsd_dir : str
	Directory of the NSD.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4',
	'PPA', 'VWFA'])
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Load the results
# =============================================================================
results_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_responses', 'zscored-'+str(args.zscore),
	'nsdsynthetic_responses_ncsnr_threshold-'+
	format(args.ncsnr_threshold, '02')+'.npy')

results = np.load(results_dir, allow_pickle=True).item()


# =============================================================================
# Load the NSD-synthetic image subclasses
# =============================================================================
labels_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsyntheticimageinformation.csv')
image_labels = pd.read_csv(labels_dir, sep=',')
unique_image_subclass_number = np.unique(image_labels['Image subclass number'])
image_subclass = []
for i in range(len(unique_image_subclass_number)):
	image_subclass.append(image_labels['Image subclass'][i*4])


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 20
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['xtick.major.size'] = 2
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 2
matplotlib.rcParams['lines.markersize'] = 2
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 1
matplotlib.rcParams['grid.alpha'] = .3


# =============================================================================
# Plot the univariate responses
# =============================================================================
# Average the univariate fMRI responses across images from the same subclass
# and across subjects
univariate_responses_avg = np.zeros((len(args.rois), len(args.subjects),
	len(image_subclass)))
for r, roi in enumerate(args.rois):
	for s, sub in enumerate(args.subjects):
		for c in range(len(unique_image_subclass_number)):
			univariate_responses_avg[r,s,c] = np.mean(
				results['univariate_responses']['s'+str(sub)+'_'+roi][c*4:c*4+4])
univariate_responses_avg = np.mean(univariate_responses_avg, 1)

# Image groups loop
groups_idx = [(0, 16), (16, 26), (26, 54), (54, 71)]
for g, img_group in enumerate(groups_idx):

	# Create the figure
	fig = plt.figure(figsize=[33, 13])

	# Plot the results
	data = univariate_responses_avg[:,img_group[0]:img_group[1]]
	cax = plt.imshow(data, cmap='cividis', vmin=0, vmax=4)

	# Colorbar
	ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
	cbar = plt.colorbar(cax, shrink=0.75, ticks=ticks,
		label='Univariate response magnitude', location='right')

	# x-axis parameters
	xticks = np.arange(data.shape[1])
	labels = image_subclass[img_group[0]:img_group[1]]
	plt.xticks(ticks=xticks, labels=labels, rotation=45)

	# y-axis parameters
	yticks = np.arange(data.shape[0])
	plt.yticks(ticks=yticks, labels=args.rois)

	# Save the figure
	file_name = 'nsdsynthetic_responses_univariate_image_group-' + str(g+1) + \
		'_ncsnr_threshold-' + format(args.ncsnr_threshold, '02') + '.svg'
	fig.savefig(file_name, dpi=300, bbox_inches='tight', format='svg')
	plt.close()


# =============================================================================
# Plot the RSMs
# =============================================================================
# Create the figure
fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
axs = np.reshape(axs, -1)

for r, roi in enumerate(args.rois):

	# Average the RSMs across subjects
	rms_sub_avg = []
	for sub in args.subjects:
		rms_sub_avg.append(results['rsms']['s'+str(sub)+'_'+roi])
	rms_sub_avg = np.mean(rms_sub_avg, 0)
	idx_diag = np.diag_indices(len(rms_sub_avg))
	rms_sub_avg[idx_diag] = 1

	# Plot the RSM
	cax = axs[r].imshow(rms_sub_avg, cmap='RdBu_r', vmin=-1, vmax=1,
		aspect='equal')

	# Colorbar
	cbar = plt.colorbar(cax, shrink=0.75, ticks=[-1, 0, 1],
		label='Pearson\'s $r$', location='right')

	# Title
	axs[r].set_title(roi)

	# Save the figure
	file_name = 'nsdsynthetic_responses_rsms_ncsnr_threshold-' + \
		format(args.ncsnr_threshold, '02') + '.svg'
	fig.savefig(file_name, dpi=300, bbox_inches='tight', format='svg')


# =============================================================================
# Plot the RSA results
# =============================================================================
# Create the figure
fig = plt.figure(figsize=(13,13))
pad = 0.5

# Plot the RSA results
rsa = results['rsa']
cax = plt.imshow(rsa, cmap='inferno', vmin=0, vmax=1, aspect='equal')

# Plot the white ROI grids
idx = np.arange(len(args.subjects), len(rsa), len(args.subjects)) - pad
for i in idx:
	plt.plot([i, i], [0-pad, len(rsa)], linewidth=2, color='white')
	plt.plot([0-pad, len(rsa)], [i, i], linewidth=2, color='white')

# Plot the diagonal cyan boxes
color = '#00FFFF'
n_sub = len(args.subjects)
idx = np.arange(0, len(rsa), n_sub) - pad
for i in idx:
	plt.plot([i, i+n_sub], [i, i], linewidth=4, color=color)
	plt.plot([i, i+n_sub], [i+n_sub, i+n_sub], linewidth=4, color=color)
	plt.plot([i, i], [i, i+n_sub], linewidth=4, color=color)
	plt.plot([i+n_sub, i+n_sub], [i, i+n_sub], linewidth=4, color=color)

# Colorbar
cbar = plt.colorbar(cax, shrink=0.75, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
	label='Pearson\'s $r$', location='right')

# Axes ticks
ticks = []
for r in range(len(args.rois)):
	idx_start = r * len(args.subjects)
	idx_end = idx_start + len(args.subjects)
	coord = (r * len(args.subjects)) + (len(args.subjects) / 2)
	ticks.append(coord - pad)
	del coord

# x-axis
plt.xticks(ticks=ticks, labels=args.rois, rotation=0)
plt.xlim(left=0-pad, right=len(rsa)-pad)

# y-axis
plt.yticks(ticks=ticks, labels=args.rois)
plt.ylim(top=0-pad, bottom=len(rsa)-pad)

# Save the figure
file_name = 'nsdsynthetic_responses_rsa_ncsnr_threshold-' + \
	format(args.ncsnr_threshold, '02') + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', format='svg')
