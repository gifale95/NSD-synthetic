"""Select NSD-core's OOD test images using k-means clustering on the fMRI
responses in MDS space.

Parameters
----------
subject : int
	Number of the used subject.
zscore : int
	Whether to z-score [1] or not [0] the fMRI responses of each vertex across
	the trials of each session.
project_dir : str
	Directory of the project folder.
nsd_dir : str
	Directory of the NSD.

"""

import argparse
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--zscore', type=int, default=0)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()

print('>>> Select NSD-core OOD images <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load NSD-core's stimulus and design information
# =============================================================================
# Load the stimulus information
data_dir = os.path.join(args.nsd_dir, 'nsddata', 'experiments', 'nsd',
	'nsd_stim_info_merged.csv')
stim_info = pd.read_csv(data_dir, delimiter=',', index_col=0)

# Load the experimental design info
nsd_expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsd', 'nsd_expdesign.mat'))
# Subtract 1 since the indices start with 1 (and not 0)
masterordering = nsd_expdesign['masterordering'] - 1
subjectim = nsd_expdesign['subjectim'] - 1


# =============================================================================
# Get the image conditions and repeats
# =============================================================================
# Completed sessions per subject
if args.subject in (1, 2, 5, 7):
	sessions = 40
elif args.subject in (3, 6):
	sessions = 32
elif args.subject in (4, 8):
	sessions = 30

# Image presentation matrix of the selected subject
image_per_session = 750
tot_images = sessions * image_per_session
img_presentation_order = subjectim[args.subject-1,masterordering[0]][:tot_images]


# =============================================================================
# Load the MDS results
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'mds_single_subjects', 'zscore-'+str(args.zscore), 'betas_mds_subject-'+
	format(args.subject, '02')+'.npy')

data = np.load(data_dir, allow_pickle=True).item()
betas_mds = data['betas_mds']


# =============================================================================
# Apply k-means clustering on the fMRI responses in MDS space
# =============================================================================
n_clusters = 15
kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=1000,
	random_state=20200220)

kmeans.fit(betas_mds)

cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_


# =============================================================================
# Define NSD-core's OOD test images
# =============================================================================
# Select as OOD cluster the cluster with largest Euclidean distance from the
# centroids of all other clusters
cluster_distance = np.zeros((n_clusters, n_clusters))
for i1 in range(n_clusters):
	for i2 in range(n_clusters):
		cluster_distance[i1,i2] = np.sum(np.sqrt(np.square(
			cluster_centers[i1] - cluster_centers[i2])))
cluster_distance = np.sum(cluster_distance, 1)
ood_cluster_label = np.argsort(cluster_distance)[::-1][0]

# Compute the Euclidean distance between the images from the OOD cluster and
# and all other images
idx_ood_image_num = np.where(labels == ood_cluster_label)[0]
idx_id_image_num = np.where(labels != ood_cluster_label)[0]
distance = np.zeros((len(idx_ood_image_num), len(idx_id_image_num)))
for i1, idx_1 in enumerate(tqdm(idx_ood_image_num)):
	for i2, idx_2 in enumerate(idx_id_image_num):
		distance[i1,i2] = np.sum(np.sqrt(np.square(
			betas_mds[idx_1] - betas_mds[idx_2])))
# Average the Euclidean distances across all ID images
distance = np.mean(distance, 1)

# Select the 284 images that are furthest away from the images of the other
# clusters
n_img = 284
idx_test_img = np.argsort(distance)[::-1][:n_img]
test_img_num_ood = data['img_num'][idx_ood_image_num[idx_test_img]]
test_img_num_ood.sort()

# Get the image repeats
test_img_ood_repeats = np.zeros(len(test_img_num_ood))
for i, img in enumerate(test_img_num_ood):
	idx = np.where(img_presentation_order == img)[0]
	test_img_ood_repeats[i] = len(idx)

# Plot the fMRI responses in MDS space
plt.figure()
plt.scatter(betas_mds[:,0], betas_mds[:,1], s=20, color='k')
for i in range(len(betas_mds)):
	if labels[i] == ood_cluster_label:
		plt.scatter(betas_mds[i,0], betas_mds[i,1], s=20, color='green')
for idx in idx_ood_image_num[idx_test_img]:
	plt.scatter(betas_mds[idx,0], betas_mds[idx,1], s=20, color='blue')
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], s=100, color='red')


# =============================================================================
# Define NSD-core's train and ID test images
# =============================================================================
# Get the number and repeats of the remaining images
np.random.shuffle(idx_id_image_num)
img_num = data['img_num'][idx_id_image_num]
# Get the repeats of the remaining images
img_repeats = np.zeros(len(img_num))
for i, img in enumerate(img_num):
	idx = np.where(img_presentation_order == img)[0]
	img_repeats[i] = len(idx)

# Randomly select 284 image conditions with most repeats for ID testing
sort_idx = np.argsort(img_repeats)[::-1]
test_img_num_id = img_num[sort_idx][:n_img]
test_img_id_repeats = img_repeats[sort_idx][:n_img]
# Sort the images based on their ID number
sort_idx_2 = np.argsort(test_img_num_id)
test_img_num_id = test_img_num_id[sort_idx_2]
test_img_id_repeats = test_img_id_repeats[sort_idx_2]

# Use the remaining images for training
train_img_num = img_num[sort_idx][n_img:]
train_img_repeats = img_repeats[sort_idx][n_img:]
# Sort the images based on their ID number
sort_idx_2 = np.argsort(train_img_num)
train_img_num = train_img_num[sort_idx_2]
train_img_repeats = train_img_repeats[sort_idx_2]


# =============================================================================
# Save the MDS results
# =============================================================================
results = {
	'train_img_num': train_img_num,
	'test_img_num_id': test_img_num_id,
	'test_img_num_ood': test_img_num_ood,
	'test_img_ood_repeats': test_img_ood_repeats,
	'test_img_id_repeats': test_img_id_repeats,
	'train_img_repeats': train_img_repeats,
	'betas_mds': betas_mds,
	'labels': labels,
	'cluster_centers': cluster_centers
	}

save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'nsdcore_train_test_splits', 'zscore-'+str(args.zscore))

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

file_name = 'nsdcore_train_test_splits_subject-' + \
	format(args.subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), results)
