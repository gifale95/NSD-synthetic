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

parser = argparse.ArgumentParser()
parser.add_argument('--nsd_dir', default='/home/ale/scratch/datasets/natural-scenes-dataset', type=str)
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
