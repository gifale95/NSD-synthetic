"""Save the NSD-core ID and OOD test images of each subject.

Parameters
----------
subject : int
	Number of the used NSD subject.
nsd_dir : str
	Directory of the NSD.
project_dir : str
	Directory of the project folder.

"""

import argparse
import numpy as np
import os
from tqdm import tqdm
import h5py
from PIL import Image


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--project_dir', default='../nsd_synthetic', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
args = parser.parse_args()

print('>>> Save test images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Access the NSD-core images
# =============================================================================
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
	'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')


# =============================================================================
# Load the train/test splits
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'nsdcore_train_test_splits', 'zscore-0',
	'nsdcore_train_test_splits_subject-'+format(args.subject, '02') + '.npy')

train_test_splits = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Save directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'nsdcore_id_ood_tests',
	'nsdcore_test_images')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)


# =============================================================================
# NSD-core ID test images
# =============================================================================
test_img_num_id = train_test_splits['test_img_num_id']

for i, img in enumerate(tqdm(test_img_num_id)):

	image = Image.fromarray(sdataset[img]).convert('RGB')

	file_name = os.path.join(save_dir, 'nsdcore_id_sub-'+
		format(args.subject, '02')+'_img-'+format(i+1, '04')+'.png')

	image.save(file_name)


# =============================================================================
# NSD-core OOD test images
# =============================================================================
test_img_num_ood = train_test_splits['test_img_num_ood']

for i, img in enumerate(tqdm(test_img_num_ood)):

	image = Image.fromarray(sdataset[img]).convert('RGB')

	file_name = os.path.join(save_dir, 'nsdcore_ood_sub-'+
		format(args.subject, '02')+'_img-'+format(i+1, '04')+'.png')

	image.save(file_name)
