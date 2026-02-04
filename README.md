# A 7T fMRI dataset of synthetic images for out-of-distribution modeling of vision

![Figure 1](figure_1.jpg)

Here we provide the code to reproduce all results from the paper:</br>
"[A 7T fMRI dataset of synthetic images for out-of-distribution modeling of vision](https://doi.org/10.48550/arXiv.2503.06286)".</br>
Alessandro T. Gifford, Radoslaw M. Cichy, Thomas Naselaris, Kendrick Kay



## 📜 Paper abstract

Large-scale visual neural datasets such as the Natural Scenes Dataset (NSD) are enabling models of the brain with performances beyond what was possible just a decade ago. However, because the stimuli of these datasets typically live within a common naturalistic visual distribution, they make it challenging to implement out-of-distribution (OOD) generalization tests crucial for the development of robust brain models. Here, we address this by releasing NSD-synthetic, a dataset of 7T fMRI responses from the same eight NSD participants for 284 synthetic images. We show that NSD-synthetic's fMRI responses reliably encode stimulus-related information and are OOD with respect to NSD; that OOD generalization tests on NSD-synthetic reveal differences between brain models that are not detected in-distribution; and that the degree of OOD (quantified as the test data distance from the training data) is predictive of the magnitude of model failures. Together, NSD-synthetic enables OOD generalization tests that facilitate the development of more robust models of visual processing.



## ♻️ Reproducibility

### 🧰 Data

The NSD dataset (including NSD-synthetic) is freely available at [http://naturalscenesdataset.org](http://naturalscenesdataset.org).



### ⚙️ Installation

This repository contains code to reproduce all paper's results.

To run the code, you first need to install the libraries in the [requirements.txt](https://github.com/gifale95/NSD-synthetic/blob/main/requirements.txt) file within an Anaconda environment. Here, we guide you through the installation steps.

First, create an [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment with the correct Python version:

```shell
conda create -n nsdsynthetic_env python=3.9
```

Next, download the [requirements.txt][requirements] file, navigate with your terminal to the download directory, and activate the Anaconda environment previously created with:

```shell
source activate nsdsynthetic_env
```

Now you can install the libraries with:

```shell
pip install -r requirements.txt
```



### 📦 Code description

* **[`00_prepare_fmri`](https://github.com/gifale95/NSD-synthetic/blob/main/00_prepare_fmri):** Prepare NSD-synthetic and NSD-core's fMRI responses for the following analyses.
* **[`paper_figure_2`](https://github.com/gifale95/NSD-synthetic/blob/main/paper_figure_2):** Analyse NSD-synthetic's univariate and multivariate fMRI responses, and noise ceiling signal-to-noise ratio (ncsnr).
* **[`paper_figure_3`](https://github.com/gifale95/NSD-synthetic/blob/main/paper_figure_3):** Perform multidimensional scaling (MDS) on NSD-synthetic and NSD-core's fMRI responses.
* **[`paper_figure_4`](https://github.com/gifale95/NSD-synthetic/blob/main/paper_figure_4):** Train encoding model on NSD-core, and test them both in-distribution (NSD-core) and out-of-distribution (NSD-synthetic).
* **[`paper_figure_5`](https://github.com/gifale95/NSD-synthetic/blob/main/paper_figure_5):** Compare diffent encoding models based on their in-distribution (NSD-core) and out-of-distribution (NSD-synthetic) performances.
* **[`paper_figure_6`](https://github.com/gifale95/NSD-synthetic/blob/main/paper_figure_6):** Compare the out-of-distribution generalization performance of encoding models tested on individual NSD-synthetic image classes.
* **[`paper_figure_7`](https://github.com/gifale95/NSD-synthetic/blob/main/paper_figure_7):** Compare the generalization performance of encoding models tested both in- and out-of-distribution on NSD-core, and out-of-distribution on NSD-synthetic.



### 🧠 Flattened cortical surface plots

In Figures 2, 4-7, we plotted results on flattened cortical surfaces using [pycortex' fsaverage subject](https://figshare.com/articles/dataset/fsaverage_subject_for_pycortex/9916166).

For visualization purposes, we manually drew surface labels based on the [“streams” ROI collection](https://cvnlab.slite.page/p/X_7BBMgghj/ROIs) as provided in the NSD data release. To use these labels, add the [`overlays.svg`](https://github.com/gifale95/NSD-synthetic/blob/main/pycortex_stream_labels/overlays.svg) file to the pycortex fsaverage subject folder (within an Anaconda environment, you should find this folder at: `../anaconda3/envs/env_name/share/pycortex/db/fsaverage`).



## ❗ Issues

If you experience problems with the code submit an issue, or get in touch with Ale (alessandro.gifford@gmail.com).



## 📜 Citation
If you use any of our data or code, please cite:

> * Gifford AT, Cichy RM, Naselaris T, Kay K. 2025. A 7T fMRI dataset of synthetic images for out-of-distribution modeling of vision. _arXiv preprint_, arXiv:2503.06286. DOI: [https://doi.org/10.48550/arXiv.2503.06286](https://doi.org/10.48550/arXiv.2503.06286)
> * Allen EJ, St-Yves G, Wu Y, Breedlove JL, Prince JS, Dowdle LT, Nau M, Caron B, Pestilli F, Charest I, Hutchinson BJ, Naselaris T, Kay K. 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. _Nature neuroscience_, 25(1), 116-126. DOI: [https://doi.org/10.1038/s41593-021-00962-x](https://doi.org/10.1038/s41593-021-00962-x)
