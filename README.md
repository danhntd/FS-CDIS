# The Art of Camouflage: Few-shot Learning for Animal Detection and Segmentation

This repository is the official implementation of the paper entitled: **The Art of Camouflage: Few-shot Learning for Animal Detection and Segmentation**
**Authors**: Thanh-Danh Nguyen , Anh-Khoa Nguyen Vu, Nhat-Duy Nguyen, Vinh-Tiep Nguyen, Thanh Duc Ngo, Thanh-Toan Do, Minh-Triet Tran, Tam V. Nguyen*.

[[Preprint]](https://arxiv.org/abs/2304.07444)


## 1. Environment Setup
Download and install Anaconda with the recommended version from [Anaconda Homepage](https://www.anaconda.com/download): [Anaconda3-2019.03-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh) 
 
```
git clone https://github.com/danhntd/FS-CDIS.git
cd FSCDIS
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

After completing the installation, please create and initiate the workspace with the specific versions below. The experiments were conducted on a Linux server with a single `GeForce RTX 2080Ti GPU`, CUDA 11.1, Torch 1.9.

```
conda create --name FSCDIS python=3
conda activate FSCDIS
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
```

This source code is based on [Detectron2](https://github.com/facebookresearch/detectron2). Please refer to INSTALL.md for the pre-built or building Detectron2 from source.

After setting up the dependencies, use the command `pip install -e .` in this root to finish.

## 2. Data Preparation


### Download the datasets

The proposed CAMO-FS is available at this [link]().

### Register datasets
Detectron2 requires a step of data registration for those who want to use the external datasets ([Detectron2 Docs](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)).

```
```

## 3. Training Pipeline
Our proposed FS-CDIS framework:
<img align="center" src="/visualization/framework.png">




Initial parameters:
```

```

### Training

```

```

### Testing

```

```

The whole script commands can be found in `./scripts/*`.

**Released checkpoints and results:**

We provide the checkpoints of our final model :

| Model R-101     | FS-CDIS-Triplet | FS-CDIS-Memory |
| ------------- |:---------------------:|:--------------------------:|
| 1-shot |   [link](https://)    |     [link](https://)       |
| 2-shot |   [link](https://)    |     [link](https://)       |
| 3-shot |   [link](https://)    |     [link](https://)       |
| 5-shot |   [link](https://)    |     [link](https://)       |


## 4. Visualization

<p align="center">
  <img width="800" src="/visualization/visualization.png">
</p>

## Citation
Please use the following bibtex to cite this repository:
```
@article{nguyen2023few,
  title={Few-shot Camouflaged Animal Detection and Segmentation},
  author={Nguyen, Thanh-Danh and Vu, Anh-Khoa Nguyen and Nguyen, Nhat-Duy and Nguyen, Vinh-Tiep and Ngo, Thanh Duc and Do, Thanh-Toan and Tran, Minh-Triet and Nguyen, Tam V},
  journal={arXiv preprint arXiv:2304.07444},
  year={2023}
}
```

## Acknowledgements

[iMTFA](https://github.com/danganea/iMTFA) [Detectron2](https://github.com/facebookresearch/detectron2.git) 