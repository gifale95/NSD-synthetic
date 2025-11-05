"""Plot the analyses results for individual NSD-synthetic image classes.

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
nsd_dir : str
	Directory of the NSD.
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
from scipy.stats import ttest_1samp
import pandas as pd
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--project_dir', default='../projects/nsd_synthetic', type=str)
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
gray = (75/255, 75/255, 75/255)
color_border = (127/255, 127/255, 127/255)

# Load the MDS results
dat_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'mds', 'zscore-'+str(args.zscore),
	'mds.npy')
mds = np.load(dat_dir, allow_pickle=True).item()

# Load the NSD-synthetic image classes
labels_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsyntheticimageinformation.csv')
image_labels = pd.read_csv(labels_dir, sep=',')
image_class = list(image_labels['Image class'])

# Create the figure
fig = plt.figure(figsize=(10, 10))

# Plot the fMRI responses for the NSD-core train images in MDS space
plt.scatter(mds['betas_mds_all_sub'][284:,0], mds['betas_mds_all_sub'][284:,1],
	s=50, color=gray, linewidths=0, alpha=.15)

# Loop across image classes
class_plot_order = ['Chromatic noise', 'Spiral gratings', 'Single words',
	'Noise', 'Phase-coherence modulation', 'Contrast modulation',
	'Manipulated scenes', 'Natural scenes']
for cl in class_plot_order:
	idx = np.asarray(
		[i for i, item in enumerate(image_class) if cl in item])

	# Get the image color
	if cl == 'Natural scenes':
		color = colors[0]
	elif cl == 'Manipulated scenes':
		color = colors[1]
	elif cl == 'Contrast modulation':
		color = colors[2]
	elif cl == 'Phase-coherence modulation':
		color = colors[3]
	elif cl == 'Noise':
		color = colors[4]
	elif cl == 'Single words':
		color = colors[5]
	elif cl == 'Spiral gratings':
		color = colors[6]
	elif cl == 'Chromatic noise':
		color = colors[7]

	# Plot the fMRI responses for the NSD-synthetic images in MDS space
	plt.scatter(mds['betas_mds_all_sub'][idx,0],
		mds['betas_mds_all_sub'][idx,1], s=100, color=color, linewidths=0.5,
		edgecolors=color_border, alpha=alpha)

	# Print MDS response distance scores
	score = np.mean(mds['image_class_euclidean_distance'][cl])
	print('\n>>> Voxel average score ' + cl + ': ' + str(score))

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

# Loop over NSD-synthetic image classes
img_classes = [
	'Natural scenes',
	'Manipulated scenes',
	'Phase-coherence modulation',
	'Single words',
	'Contrast modulation',
	'Spiral gratings',
	'Noise',
	'Chromatic noise'
	]
for cl in img_classes:

	# Get the class image number
	if cl == 'Chromatic noise':
		class_num = 0
	if cl == 'Contrast modulation':
		class_num = 1
	if cl == 'Manipulated scenes':
		class_num = 2
	if cl == 'Natural scenes':
		class_num = 3
	if cl == 'Noise':
		class_num = 4
	if cl == 'Phase-coherence modulation':
		class_num = 5
	if cl == 'Single words':
		class_num = 6
	if cl == 'Spiral gratings':
		class_num = 7

	# Get the explained variance scores
	expl_var = []
	for s, sub in enumerate(args.subjects):
		# Load the encoding accuracy results
		data_dir = os.path.join(args.project_dir, 'results',
			'nsdsynthetic_image_classes', 'encoding_accuracy', 'zscore-'+
			str(args.zscore), 'model-'+args.model,
			'encoding_accuracy_nsdsynthetic_sub-'+format(sub, '02')+
			'_image_class-'+str(class_num)+'.npy')
		results = np.load(data_dir, allow_pickle=True).item()
		# Load the explained variance scores
		lh_data = results['lh_explained_variance']
		rh_data = results['rh_explained_variance']
		# Remove vertices with noise ceiling values below a threshold, since
		# they cannot be interpreted in terms of modeling
		lh_idx = results['lh_nc_all_nsdsynthetic_image_classes'] > 0.3
		rh_idx = results['rh_nc_all_nsdsynthetic_image_classes'] > 0.3
		lh_data[~lh_idx] = np.nan
		rh_data[~rh_idx] = np.nan
		# Store the data
		expl_var.append(np.append(lh_data, rh_data))
		del results

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
	file_name = 'explained_variance_model-' + args.model + '_' + \
		cl.replace(' ', '_') + '.svg'
	fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
		format='svg')
	plt.close()

	# Print the mean encoding accuracy
	print('\n>>> Voxel average score ' + cl + ': ' + str(np.nanmean(expl_var)))


# =============================================================================
# Plot the image identification accuracy results
# =============================================================================
# Plot parameters
alpha_chance = .5
alpha_ci = 0.2
fontsize = 68
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

# Load the identification results
data_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'image_identification_accuracy', 'zscore-'+
	str(args.zscore), 'model-'+args.model, 'image_identification_accuracy.npy')
results = np.load(data_dir, allow_pickle=True).item()

# Create the figure
fig = plt.figure(figsize=(20, 15))

# Loop over image classes (sort the image classes by accuracy)
img_classes = sorted(results['mean_rank_nsdsynthetic_classes'],
	key=results['mean_rank_nsdsynthetic_classes'].get)
scores_mean = []
for i, cl in enumerate(img_classes):

	# Add 1 so that the rankings start from 1 (now they start from 0)
	scores_rank_nsdsynthetic_classes = \
		results['scores_rank_nsdsynthetic_classes'][cl] + 1
	mean_rank_nsdsynthetic_classes = \
		results['mean_rank_nsdsynthetic_classes'][cl] + 1

	# Get the image color
	if cl == 'Natural scenes':
		color = colors[0]
	elif cl == 'Manipulated scenes':
		color = colors[1]
	elif cl == 'Contrast modulation':
		color = colors[2]
	elif cl == 'Phase-coherence modulation':
		color = colors[3]
	elif cl == 'Noise':
		color = colors[4]
	elif cl == 'Single words':
		color = colors[5]
	elif cl == 'Spiral gratings':
		color = colors[6]
	elif cl == 'Chromatic noise':
		color = colors[7]

	# Plot the chance lines
	plt.plot([i-0.25, i+0.25], [32, 32], '--k', linewidth=2,
		alpha=alpha_chance, zorder=1)

	# Plot the image identification accuracies
	parts = plt.violinplot(
		dataset=scores_rank_nsdsynthetic_classes,
		positions=[i],
		widths=0.75,
		showmeans=False,
		showextrema=False,
		points=900
		)

	# Set the violinplot colors
	for pc in parts['bodies']:
		pc.set_facecolor(color)
		pc.set_edgecolor(None)
		pc.set_alpha(.5)

	# Plot the individual data points
	color_points = [(200/255, 200/255, 200/255)]
	x = np.random.normal(loc=i, scale=0.1, size=len(scores_rank_nsdsynthetic_classes))
	plt.scatter(x, scores_rank_nsdsynthetic_classes, s=100, color=color,
		alpha=0.1, edgecolors='k', linewidths=1, zorder=2)

	# Plot the mean
	x = i
	plt.scatter(x, mean_rank_nsdsynthetic_classes, s=500, color=color, alpha=1,
		edgecolors='k', linewidths=2, zorder=2)
	scores_mean.append(int(np.round(mean_rank_nsdsynthetic_classes)))
	plt.text(x, mean_rank_nsdsynthetic_classes+2, scores_mean[i], ha='center')

# y-axis
ticks = [1, 16, 32, 48, 64]
labels = [1, 16, 32, 48, 64]
plt.yticks(ticks=ticks, labels=labels)
plt.ylabel('Correct image rank', fontsize=fontsize)
plt.ylim(bottom=1, top=64)

# x-axis
ticks = np.arange(len(img_classes))
labels = []
plt.xticks(ticks=ticks, labels=labels, fontsize=fontsize, rotation=0)
#plt.setp(plt.gca().get_xticklabels(), ha='right') # Right align x-labels
plt.xlabel('NSD-synthetic image classes', fontsize=fontsize)
plt.xlim(left=-0.6, right=7.6)

# Save the figure
file_name = 'identification_accuracy_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()


# =============================================================================
# Plot the relationship between response distance scores, encoding
# accuracy scores, and identification ranks of each NSD-synthetic image class
# =============================================================================
# Plot parameters
fontsize = 68
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
color_border = (127/255, 127/255, 127/255)

# Load the NSD-synthetic image classes
labels_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsdsynthetic', 'nsdsyntheticimageinformation.csv')
image_labels = pd.read_csv(labels_dir, sep=',')

# NSD-synthetic image classes
img_classes = [
	'Natural scenes',
	'Manipulated scenes',
	'Phase-coherence modulation',
	'Single words',
	'Contrast modulation',
	'Spiral gratings',
	'Noise',
	'Chromatic noise'
	]

# Load the MDS response distance scores
data_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'mds', 'zscore-'+str(args.zscore),
	'mds.npy')
data = np.load(data_dir, allow_pickle=True).item()['image_class_euclidean_distance']
mds = np.zeros((len(args.subjects), len(img_classes)))
for i, cl in enumerate(img_classes):
	mds[:,i] = data[cl]

# Load the encoding accuracy scores
encoding = np.zeros((len(args.subjects), len(img_classes)))
for i, cl in enumerate(img_classes):
	# Get the class image number
	if cl == 'Chromatic noise':
		class_num = 0
	if cl == 'Contrast modulation':
		class_num = 1
	if cl == 'Manipulated scenes':
		class_num = 2
	if cl == 'Natural scenes':
		class_num = 3
	if cl == 'Noise':
		class_num = 4
	if cl == 'Phase-coherence modulation':
		class_num = 5
	if cl == 'Single words':
		class_num = 6
	if cl == 'Spiral gratings':
		class_num = 7
	# Get the explained variance scores
	for s, sub in enumerate(args.subjects):
		# Load the encoding accuracy results
		data_dir = os.path.join(args.project_dir, 'results',
			'nsdsynthetic_image_classes', 'encoding_accuracy', 'zscore-'+
			str(args.zscore), 'model-'+args.model,
			'encoding_accuracy_nsdsynthetic_sub-'+format(sub, '02')+
			'_image_class-'+str(class_num)+'.npy')
		results = np.load(data_dir, allow_pickle=True).item()
		# Load the explained variance scores
		lh_data = results['lh_explained_variance']
		rh_data = results['rh_explained_variance']
		# Remove vertices with noise ceiling values below a threshold, since
		# they cannot be interpreted in terms of modeling
		lh_idx = results['lh_nc_all_nsdsynthetic_image_classes'] > 0.3
		rh_idx = results['rh_nc_all_nsdsynthetic_image_classes'] > 0.3
		lh_data[~lh_idx] = np.nan
		rh_data[~rh_idx] = np.nan
		# Store the data
		encoding[s,i] = np.nanmean(np.append(lh_data, rh_data))

# Load the image identification scores
data_dir = os.path.join(args.project_dir, 'results',
	'nsdsynthetic_image_classes', 'image_identification_accuracy', 'zscore-'+
	str(args.zscore), 'model-'+args.model, 'image_identification_accuracy.npy')
data = np.load(data_dir, allow_pickle=True).item()
identification = np.zeros((len(args.subjects), len(img_classes)))
for i, cl in enumerate(img_classes):
	class_img_num = \
		[c for c, item in enumerate(list(image_labels['Image class'])) if item == cl]
	class_img_num = np.asarray(class_img_num)
	identification[:,i] = np.mean(data['rank_nsdsynthetic'][:,class_img_num], 1)

# Correlation between MDS response distance scores and encoding accuracy scores
corr = np.zeros((len(args.subjects)))
for s in range(len(args.subjects)):
	corr[s] = spearmanr(mds[s], encoding[s])[0]
pval = ttest_1samp(corr, 0, alternative='less')[1]
sig = True if pval < (0.05 / 3) else False # Bonferroni correction across the 3 comparisons
print('\n>>> Correlation between MDS response distance and encoding accuracy scores')
print('>>> Average correaltion score: ' + str(np.mean(corr)))
print('>>> p-value: ' + str(np.mean(pval)))
print('>>> Significant: ' + str(sig))
# Plot the correlation as a scatterplot
fig = plt.figure(figsize=(15, 15))
# Plot the regression line
x = np.mean(mds, 0)
y = np.mean(encoding, 0)
m, b = np.polyfit(x, y, 1) # slope, intercept
x = np.arange(225, 625)
plt.plot(x, m*x + b, linewidth=3, color=color_border, zorder=1)
# Loop over image classes (sort the image classes by accuracy)
for i, cl in enumerate(img_classes):
	# Get the image color
	if cl == 'Natural scenes':
		color = colors[0]
	elif cl == 'Manipulated scenes':
		color = colors[1]
	elif cl == 'Contrast modulation':
		color = colors[2]
	elif cl == 'Phase-coherence modulation':
		color = colors[3]
	elif cl == 'Noise':
		color = colors[4]
	elif cl == 'Single words':
		color = colors[5]
	elif cl == 'Spiral gratings':
		color = colors[6]
	elif cl == 'Chromatic noise':
		color = colors[7]
	# Plot the results on scatterplots (single subjects)
	plt.scatter(mds[:,i], encoding[:,i], s=1500, color=color, linewidths=0,
		edgecolors=color_border, alpha=.25, zorder=2)
	# Plot the results on scatterplots (subject average)
	plt.scatter(np.mean(mds[:,i]), np.mean(encoding[:,i]), s=3000, color=color,
		linewidths=0, edgecolors='k', alpha=1, zorder=3)
# y-axis
ticks = [20, 40, 60, 80, 100]
labels = [20, 40, 60, 80, 100]
plt.yticks(ticks=ticks, labels=labels)
plt.ylabel('Noise-ceiling-normalized\nencoding accuracy (%)',
	fontsize=fontsize)
plt.ylim(bottom=0, top=60)
# x-axis
ticks = [175, 425, 675]
labels = [175, 425, 675]
plt.xticks(ticks=ticks, labels=labels, fontsize=fontsize, rotation=0)
plt.xlabel('Euclidean response distance', fontsize=fontsize)
plt.xlim(left=175, right=675)
# Save the figure
file_name = 'response_distance_vs_encoding_accuracy_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()

# Correlation between MDS response distance scores and identification ranks
corr = np.zeros((len(args.subjects)))
for s in range(len(args.subjects)):
	corr[s] = spearmanr(mds[s], identification[s])[0]
pval = ttest_1samp(corr, 0, alternative='greater')[1]
sig = True if pval < (0.05 / 3) else False # Bonferroni correction across the 3 comparisons
print('\n>>> Correlation between MDS response distance and image identification ranks')
print('>>> Average correaltion score: ' + str(np.mean(corr)))
print('>>> p-value: ' + str(np.mean(pval)))
print('>>> Significant: ' + str(sig))
# Plot the correlation as a scatterplot
fig = plt.figure(figsize=(15, 15))
# Plot the regression line
x = np.mean(mds, 0)
y = np.mean(identification, 0)
m, b = np.polyfit(x, y, 1) # slope, intercept
x = np.arange(225, 625)
plt.plot(x, m*x + b, linewidth=3, color=color_border, zorder=1)
# Loop over image classes (sort the image classes by accuracy)
for i, cl in enumerate(img_classes):
	# Get the image color
	if cl == 'Natural scenes':
		color = colors[0]
	elif cl == 'Manipulated scenes':
		color = colors[1]
	elif cl == 'Contrast modulation':
		color = colors[2]
	elif cl == 'Phase-coherence modulation':
		color = colors[3]
	elif cl == 'Noise':
		color = colors[4]
	elif cl == 'Single words':
		color = colors[5]
	elif cl == 'Spiral gratings':
		color = colors[6]
	elif cl == 'Chromatic noise':
		color = colors[7]
	# Plot the results on scatterplots (single subjects)
	plt.scatter(mds[:,i], identification[:,i], s=1500, color=color, linewidths=0,
		edgecolors=color_border, alpha=.25, zorder=2)
	# Plot the results on scatterplots (subject average)
	plt.scatter(np.mean(mds[:,i]), np.mean(identification[:,i]), s=3000,
		color=color, linewidths=0, edgecolors=color_border, alpha=1, zorder=3)
# y-axis
ticks = [10, 20, 30, 40, 50]
labels = [10, 20, 30, 40, 50]
plt.yticks(ticks=ticks, labels=labels)
plt.ylabel('Correct image rank', fontsize=fontsize)
plt.ylim(bottom=1, top=30)
# x-axis
ticks = [175, 425, 675]
labels = [175, 425, 675]
plt.xticks(ticks=ticks, labels=labels, fontsize=fontsize, rotation=0)
plt.xlabel('Euclidean response distance', fontsize=fontsize)
plt.xlim(left=175, right=675)
# Save the figure
file_name = 'response_distance_vs_identification_model-' + args.model + '.svg'
fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True,
	format='svg')
plt.close()

# Correlation between encoding accuracy scores and identification ranks
corr = np.zeros((len(args.subjects)))
for s in range(len(args.subjects)):
	corr[s] = spearmanr(encoding[s], identification[s])[0]
pval = ttest_1samp(corr, 0, alternative='less')[1]
sig = True if pval < (0.05 / 3) else False # Bonferroni correction across the 3 comparisons
print('\n>>> Correlation between encoding accuracy scores and image identification ranks')
print('>>> Average correaltion score: ' + str(np.mean(corr)))
print('>>> p-value: ' + str(np.mean(pval)))
print('>>> Significant: ' + str(sig))
