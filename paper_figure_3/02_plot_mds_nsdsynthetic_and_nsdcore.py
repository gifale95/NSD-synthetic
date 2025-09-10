"""Plot the results of multidimensional scaling (MDS) applied jointly on
NSD-synthetic and NSD-core's first two scan sessions.

Parameters
----------
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept vertices: only vertices above this
	threshold are used.
zscore : int
	Whether to betas were z-score [1] or not [0] prior to applying MDS.
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
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Load the MDS results
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'mds_nsdsynthetic_nsdcore',
	'mds_zscore-' + str(args.zscore) + '_ncsnr_threshold-' + \
		str(args.ncsnr_threshold) + '.npy')
results = np.load(data_dir, allow_pickle=True).item()

betas_mds = results['betas_mds']
min_betas = min(betas_mds.flatten())
max_betas = max(betas_mds.flatten())


# =============================================================================
# Load the NSD-synthetic image classes
# =============================================================================
labels_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsyntheticimageinformation.csv')
image_labels = pd.read_csv(labels_dir, sep=',')
image_class = list(image_labels['Image class'])


# =============================================================================
# Load the NSD-synthetic trials conditions
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas', 'zscore-'+
	str(args.zscore), 'sub-01')
masterordering = np.load(os.path.join(data_dir, 'meatadata_nsdsynthetic.npy'),
	allow_pickle=True).item()['masterordering']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 15
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(221/255, 204/255, 119/255), (204/255, 102/255, 119/255)]
colors = [
	(85/255, 220/255, 255/255), # Light blue
	(221/255, 0/255, 50/255), # Red
	(245/255, 126/255, 0/255), # Orange
	(60/255, 215/255, 0/255), # Green
	(255/255, 191/255, 204/255), # Pink
	(127/255, 0/255, 127/255), # Purple
	(0/255, 127/255, 127/255), # Teal
	(255/255, 214/255, 0/255), # Gold
	]


# =============================================================================
# Plot the MDS results
# =============================================================================
# Create the figure
fig = plt.figure(figsize=(13, 13))

# Loop across NSD-synthetic images
for i in reversed(list(range(744))):

	# Get the trial image class number
	class_name = image_class[masterordering[i]]

	# Get image border color
	if class_name == 'Natural scenes':
		color = colors[0]
	elif class_name == 'Manipulated scenes':
		color = colors[1]
	elif class_name == 'Contrast modulation':
		color = colors[2]
	elif class_name == 'Phase-coherence modulation':
		color = colors[3]
	elif class_name == 'Noise':
		color = colors[4]
	elif class_name == 'Single words':
		color = colors[5]
	elif class_name == 'Spiral gratings':
		color = colors[6]
	elif class_name == 'Chromatic noise':
		color = colors[7]

	# Plot the fMRI responses for the NSD-synthetic trials in MDS space
	plt.scatter(betas_mds[i,0], betas_mds[i,1],	s=50, color=color,
		linewidths=0, alpha=1)

# Plot the fMRI responses for the NSD-core trials in MDS space
plt.scatter(betas_mds[744:1488,0], betas_mds[744:1488,1],
	s=50, color='k', linewidths=0, alpha=1)
plt.scatter(betas_mds[1488:,0], betas_mds[1488:,1],
	s=50, color='gray', linewidths=0, alpha=1)

# x-axis
plt.xticks([])
#plt.xlim(left=min_betas, right=max_betas)

# y-axis
plt.yticks([])
#plt.ylim(bottom=min_betas, top=max_betas)


# =============================================================================
# Save the figure
# =============================================================================
file_name = 'mds_nsdsynthetic_nsdcore_zscore-' + str(args.zscore) + \
	'_ncsnr_threshold-' + str(args.ncsnr_threshold) + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', format='svg')
