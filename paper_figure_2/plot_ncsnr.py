"""Plot NSD-synthetic and NSD-core's NCSNR.

Parameters
----------
subjects : list
	List with all used NSD subject.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Load NSD-synthetic's NCSNR
# =============================================================================
lh_ncsnr_nsdsynthetic = []
rh_ncsnr_nsdsynthetic = []

for sub in args.subjects:

	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas',
		'zscore-'+str(args.zscore), 'sub-0'+format(sub),
		'meatadata_nsdsynthetic.npy')
	metadata = np.load(data_dir, allow_pickle=True).item()
	lh_ncsnr_nsdsynthetic.append(metadata['lh_ncsnr'])
	rh_ncsnr_nsdsynthetic.append(metadata['rh_ncsnr'])


# =============================================================================
# Load NSD-core's NCSNR
# =============================================================================
lh_ncsnr_nsdcore = []
rh_ncsnr_nsdcore = []

for sub in args.subjects:

	data_dir = os.path.join(args.project_dir, 'results',
		'train_test_session_control-0', 'fmri_betas', 'zscore-'+
		str(args.zscore), 'sub-0'+format(sub), 'meatadata_nsdcore.npy')
	metadata = np.load(data_dir, allow_pickle=True).item()
	lh_ncsnr_nsdcore.append(metadata['lh_ncsnr'])
	rh_ncsnr_nsdcore.append(metadata['rh_ncsnr'])


# =============================================================================
# Plot parameters for colorbar
# =============================================================================
subject = 'fsaverage'
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'


# =============================================================================
# Plot NSD-synthetic's NCSNR
# =============================================================================
for s, sub in enumerate(args.subjects):

	# Create the vertex data
	data = np.append(lh_ncsnr_nsdsynthetic[s], rh_ncsnr_nsdsynthetic[s])
	vertex_data = cortex.Vertex(data, subject, cmap='hot', vmin=0, vmax=2,
		with_colorbar=True)

	# Plot the figure
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
	fig.savefig('ncsnr_nsdsynthetic_sub-0'+str(sub)+'.svg', dpi=300,
		bbox_inches='tight', transparent=True, format='svg')
	plt.close()


# =============================================================================
# Plot NSD-core's NCSNR
# =============================================================================
for s, sub in enumerate(args.subjects):

	# Create the vertex data
	data = np.append(lh_ncsnr_nsdcore[s], rh_ncsnr_nsdcore[s])
	vertex_data = cortex.Vertex(data, subject, cmap='hot', vmin=0, vmax=2,
		with_colorbar=True)

	# Plot the figure
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
	fig.savefig('ncsnr_nsdcore_sub-0'+str(sub)+'.svg', dpi=300,
		bbox_inches='tight', transparent=True, format='svg')
	plt.close()
