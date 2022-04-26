# Palette: Image-to-Image Diffusion Models

[Paper](https://arxiv.org/pdf/2111.05826.pdf ) |  [Project](https://iterative-refinement.github.io/palette/ )

## Brief

This is an unofficial implementation of **Palette: Image-to-Image Diffusion Models** by **Pytorch**, and it is mainly inherited from its super-resolution version [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). 

There are some implement details with paper description, which may be different from the actual `Palette` structure.

- We adapted the U-Net architecture described in  `Guided-Diffusion`.
- We used the attention mechanism in low-resolution features (16×16) like vanilla `DDPM`.
- We encode the $\gamma$ rather than $t$ in `Palette` and embed it with affine transformation.

## Status

### Code
- [x] Diffusion Model Pipeline
- [x] Train/Test Process
- [x] Save/Load Training State
- [x] Logger/Tensorboard
- [x] DDP Training
- [x] EMA
- [x] Dataset (now just for inpainting)
- [ ] Metrics

  


### Task

I try to finish following tasks in order. 

- [ ] Inpainting on CelebaHQ with 128×128 center mask
- [ ] Inpainting on Places2 with 128×128 center mask
- [ ] Uncropping on Places2
- [ ] Colorization on ImageNet val set 

## Results

### Visuals

Coming soon.

### Metrics

| Tasks/Metrics        | FID(-) | IS(+) | CA(+) | PD(-) |
| -------------------- | ----------- | -------- | ---- | ---- |



## Usage

### Environment
```python
pip install -r requirement.txt
```

### Pre-trained Model

### Data Prepare

[Places2](http://places2.csail.mit.edu/download.html) | [ImageNet](https://www.image-net.org/download.php)



### Training/Resume Training

### Test/Evaluation


## Acknowledge

Our work is based on the following theoretical works:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Palette: Image-to-Image Diffusion Models](https://arxiv.org/pdf/2111.05826.pdf)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

and we are benefiting a lot from the following projects:

- [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
- [LouisRouss/Diffusion-Based-Model-for-Colorization](https://github.com/LouisRouss/Diffusion-Based-Model-for-Colorization)

Code template from my another seed project: [distributed-pytorch-template](https://github.com/Janspiry/distributed-pytorch-template)