"""Plot the results of multidimensional scaling (MDS) applied on NSD-synthetic.

Parameters
----------
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
nsd_dir : str
	Directory of the NSD.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

parser = argparse.ArgumentParser()
parser.add_argument('--ncsnr_threshold', type=float, default=0.6)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
args = parser.parse_args()


# =============================================================================
# Load the MDS results
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'mds_nsdsynthetic',
	'zscored-'+str(args.zscore), 'mds_nsdsynthetic_ncsnr_threshold-'+
	format(args.ncsnr_threshold, '02')+'.npy')
results = np.load(data_dir, allow_pickle=True).item()
betas_mds = results['betas_mds']


# =============================================================================
# Load the NSD-synthetic image classes
# =============================================================================
labels_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsyntheticimageinformation.csv')
image_labels = pd.read_csv(labels_dir, sep=',')
image_class = list(image_labels['Image class'])


# =============================================================================
# Read all images into memory
# =============================================================================
# Stimuli
stimuli = []
stimuli_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
	'nsdsynthetic', 'nsdsynthetic_stimuli.hdf5')
sf = h5py.File(stimuli_dir, 'r')
sdataset = sf.get('imgBrick')
for img in tqdm(sdataset):
	img = (np.sqrt(img/255)*255).astype(np.uint8)
	img = Image.fromarray(img).convert('RGB')
	width, height = img.size
	new_side = min(width, height)
	left = (width - new_side) // 2
	top = (height - new_side) // 2
	right = left + new_side
	bottom = top + new_side
	img = img.crop((left, top, right, bottom))
	img = np.array(img) / 255
	stimuli.append(img)
	del img

# Colorstimuli
colorstimuli = []
stimuli_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
	'nsdsynthetic', 'nsdsynthetic_colorstimuli_subj01.hdf5')
sf = h5py.File(stimuli_dir, 'r')
sdataset = sf.get('imgBrick')
for img in tqdm(sdataset):
	img = (np.sqrt(img/255)*255).astype(np.uint8)
	img = Image.fromarray(img).convert('RGB')
	width, height = img.size
	new_side = min(width, height)
	left = (width - new_side) // 2
	top = (height - new_side) // 2
	right = left + new_side
	bottom = top + new_side
	img = img.crop((left, top, right, bottom))
	img = np.array(img) / 255
	colorstimuli.append(img)
	del img


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 30
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
colors = [
	(0.0, 1.0, 1.0), # Cyan
	(1.0, 0.0, 0.0), # Red
	(1.0, 0.65, 0.0), # Orange
	(0.75, 1.0, 0.0), # Lime
	(1.0, 0.75, 0.8), # Pink
	(0.5, 0.0, 0.5), # Purple
	(0.0, 0.5, 0.5), # Teal
	(1.0, 0.84, 0.0), # Gold
	]


# =============================================================================
# Plot the MDS results
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 13))

# Plot the NSD-synthetic images in MDS space
class_plot_order = ['Chromatic noise', 'Spiral gratings', 'Single words',
	'Noise', 'Phase-coherence modulation', 'Contrast modulation',
	'Manipulated scenes', 'Natural scenes']

# Loop across image classes
for class_name in class_plot_order:
	idx = [i for i, item in enumerate(image_class) if class_name in item]

	# Get the image border color
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

	# Loop across single images
	for i in idx:

		# Get the image
		if i < 220:
			img = stimuli[i]
		else:
			img = colorstimuli[i-220]

		# Plot the image
		imagebox = OffsetImage(img, zoom=0.04) # Adjust zoom as needed
		ab = AnnotationBbox(
			imagebox,
			(betas_mds[i,0], betas_mds[i,1]),
			frameon=True,
			pad=0,
			bboxprops=dict(edgecolor=color, linewidth=4))
		ax.add_artist(ab)

# x-axis
plt.xticks([])
plt.xlim(left=min(betas_mds[:,0])-5, right=max(betas_mds[:,0])+5)

# y-axis
plt.yticks([])
plt.ylim(bottom=min(betas_mds[:,1]-5), top=max(betas_mds[:,1])+5)

# Save the figure
file_name = 'mds_nsdsynthetic_all_subjects_and_vertices_zscored-' + \
	str(args.zscore) + '_ncsnr_threshold-' + \
	format(args.ncsnr_threshold, '02') + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', format='svg')
