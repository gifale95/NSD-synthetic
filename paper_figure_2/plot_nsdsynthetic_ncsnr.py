"""Plot NSD-synthetic's ncsnr.

Parameters
----------
subjects : list
	List with all used NSD subject.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import cortex
import cortex.polyutils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Load the ncsnr
# =============================================================================
lh_ncsnr_nsdsynthetic = []
rh_ncsnr_nsdsynthetic = []

for sub in args.subjects:

	data_dir = os.path.join(args.project_dir, 'results', 'fmri_betas', 'sub-0'+
		format(sub), 'meatadata_nsdsynthetic.npy')
	metadata = np.load(data_dir, allow_pickle=True).item()
	lh_ncsnr_nsdsynthetic.append(metadata['lh_ncsnr'])
	rh_ncsnr_nsdsynthetic.append(metadata['rh_ncsnr'])


# =============================================================================
# Plot parameters for colorbar
# =============================================================================
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)


# =============================================================================
# Plot the ncsnr
# =============================================================================
for s, sub in enumerate(args.subjects):

	# Create the vertex data
	subject = 'fsaverage'
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
	fig.savefig('ncsnr_nsdsynthetic_sub-0'+str(sub+1)+'.svg', dpi=300,
		bbox_inches='tight', transparent=True, format='svg')
	plt.close()
