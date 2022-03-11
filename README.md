# Palette: Image-to-Image Diffusion Models

[Paper](https://arxiv.org/pdf/2111.05826.pdf ) |  [Project](https://iterative-refinement.github.io/palette/ )

## Brief

This is a unoffical implementation about **Palette: Image-to-Image Diffusion Models** by **Pytorch**, and it is mainly inherited from its super-resolution version [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). 

## Status

I will try my best to finish following tasks in order. Welcome to any contributions for more extensive experiments and code enhancements.

- [ ] Inpainting on Places2 with 128×128 center mask
- [ ] Uncropping on Places2
- [ ] Colorization on ImageNet val set 


## Results

| Tasks/Metrics        | FID(-) | IS(+) | CA(+) | PD(-) |
| -------------------- | ----------- | -------- | ---- | ---- |



## Usage

### Environment
```python
pip install -r requirement.txt
```

### Pretrained Model

### Data Prepare

[Places2](http://places2.csail.mit.edu/download.html) | [ImageNet](https://www.image-net.org/download.php)



### Training/Resume Training

### Test/Evaluation


## Acknowledge

Our work is based on the following theoretical works:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- [Palette: Image-to-Image Diffusion Models](https://arxiv.org/pdf/2111.05826.pdf)