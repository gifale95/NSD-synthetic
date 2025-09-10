"""Create Pycortex' stream labels based on NSD's stream ROIs.

Parameters
----------
nsd_dir : str
	Directory of the NSD.

"""

import argparse
import os
import numpy as np
import nibabel as nib
import cortex
import cortex.polyutils
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()


# =============================================================================
# Create the stream labels
# =============================================================================
# Load the NSD stream ROIs
streams_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'fsaverage',
	'label')
lh_streams = np.squeeze(nib.load(
	os.path.join(streams_dir, 'lh.streams.mgz')).get_fdata())
rh_streams = np.squeeze(nib.load(
	os.path.join(streams_dir, 'rh.streams.mgz')).get_fdata())

# Prepare the data in Pycortex format
data = np.append(lh_streams, rh_streams)
data[data==0] = np.nan
subject = 'fsaverage_nsd'
vertex_data = cortex.Vertex(data, subject)

# Create the ROI labels: https://gallantlab.org/pycortex/generated/cortex.utils.add_roi.html
cortex.utils.add_roi(vertex_data, name='streams')

# Then manually draw the ROI labels using Inkscape paths.


# =============================================================================
# Plot the stream labels
# =============================================================================
# Create the vertex data
subject = 'fsaverage'
data = np.empty(327684)
data[:] = np.nan
vertex_data = cortex.Vertex(data, subject)

# Plot the vertex data
# https://gallantlab.org/pycortex/generated/cortex.quickflat.make_figure.html#cortex.quickflat.make_figure
fig = cortex.quickshow(vertex_data,
#	height=500, # Increase resolution of map and ROI contours
	with_curvature=True,
	curvature_brightness=0.5,
	with_rois=False,
	with_labels=False,
	linewidth=5,
	linecolor=(1, 1, 1),
	with_colorbar=False
	)

# Fill each ROIs region with its color (and alpha)
rois = ['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal']
colors = [(67/255, 147/255, 195/255), (209/255, 230/255, 241/255),
	(253/255, 219/255, 199/255), (214/255, 96/255, 77/255),
	(103/255, 0/255, 31/255)]
for r, roi in enumerate(rois):
	_ = cortex.quickflat.composite.add_rois(fig, vertex_data,
		roi_list=[roi], # (This defaults to all rois if not specified)
		with_labels=False,
		linewidth=5,
		linecolor=(1, 1, 1),
		labelcolor=(0.9, 0.5, 0.5),
		labelsize=20,
		roifill=colors[r],
		fillalpha=1,
#		dashes=(5, 3) # Dash length & gap btw dashes
		)
plt.show()

# Save the figure
fig.savefig('visual_streams.svg', dpi=300, bbox_inches='tight',
	transparent=True, format='svg')
