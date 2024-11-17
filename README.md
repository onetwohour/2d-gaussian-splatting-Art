# 2D Gaussian Splatting with Nerf-Art Idea

### 2D Gaussian Splatting

[Project page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/pdf/2403.17888) | [Video](https://www.youtube.com/watch?v=oaHCtB6yiKU) | [Surfel Rasterizer (CUDA)](https://github.com/hbb1/diff-surfel-rasterization) | [Surfel Rasterizer (Python)](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing) | [DTU+COLMAP (3.5GB)](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) | [SIBR Viewer Pre-built for Windows](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing) | [Web Viewer](https://github.com/mkkellogg/GaussianSplats3D) <br>

### Nerf-Art

[Project Page](https://cassiepython.github.io/nerfart/index.html) | [Paper (ArXiv)](https://arxiv.org/abs/2212.08070) | [Paper (TVCG)](https://arxiv.org/abs/2212.08070) | [Download Models and Configs](https://portland-my.sharepoint.com/:f:/g/personal/cwang355-c_my_cityu_edu_hk/EuCfP0A3BJNItp72U_FxAicBXw_kXbcKgpaqUkeZBIwgQw?e=Vnvi48)


This is a simple toy project that combines 2D Gaussian Splatting and Nerf-Art to creatively reconstruct 2D images into 3D space. It allows you to render high-resolution images quickly and apply artistic transformations in 3D space while preserving depth and different artistic styles.

> Note: This is a toy project for experimentation and learning, not intended for production use.

### Tech Stack

- 2D Gaussian Splatting: This technique breaks 2D images into small Gaussian splats, which are placed in 3D space to generate high-resolution renderings. It allows for fast and high-quality rendering.

- Nerf-Art: Using Neural Radiance Fields (NeRF), this method applies artistic style transformations to the images, allowing you to modify the atmosphere of the 3D scene and apply various artistic effects.

### Features

- 2D Gaussian Splatting: Convert 2D images into multiple Gaussian splats and place them in 3D space to generate high-quality renders quickly.

- Nerf-Art Style Transfer: Apply artistic styles to 3D data. This allows you to transform the mood of the image by integrating an artistic style into the 3D space.

- Combined Rendering: Merge 2D Gaussian Splatting and Nerf-Art, enabling you to apply artistic style transformations while maintaining natural depth in 3D space.

### Installation
---
#### Requirements
- Python 3.8
- CUDA (for GPU usage)

#### Installation Steps
1. Clone the repository:
```
git clone https://github.com/onetwohour/2d-gaussian-splatting-Art.git --recursive
cd 2d-gaussian-splatting-Art
```

2. Set up CUDA (for GPU usage):

```
# if use CUDA 11.7
pip install torch==2.0.0+cu117 torchaudio==2.0.0 torchvision==0.15.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install the dependencies:

```
pip install ninja
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

4. Train

```
python train.py -s COLMAP_PATH --config configs/vangogh.yaml
```
