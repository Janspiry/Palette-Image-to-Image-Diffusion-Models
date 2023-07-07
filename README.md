# Palette: Image-to-Image Diffusion Models

[Paper](https://arxiv.org/pdf/2111.05826.pdf ) |  [Project](https://iterative-refinement.github.io/palette/ )

## Brief

This project is an unofficial implementation of **Palette: Image-to-Image Diffusion Models** as a Python library. The engine is taken directly from Liangwei Jiang's repository [Palette-Image-to-Image-Diffusion-Models] (https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models). This runs on **Pytorch**, and the code is mainly inherited from the author's other repositories:[Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) and the template seed project [distributed-pytorch-template](https://github.com/Janspiry/distributed-pytorch-template).

Some details of the upstream implementation:

> - We adapted the U-Net architecture used in  `Guided-Diffusion`, which give a substantial boost to sample quality.
> - We used the attention mechanism in low-resolution features (16×16) like vanilla `DDPM`.
> - We encode the $\gamma$ rather than $t$ in `Palette` and embed it with affine transformation.
> - We fix the variance $Σ_\theta(x_t, t)$ to a constant during the inference as described in `Palette`.

See the sources for more theoretical and implementational details of the models.

## Usage

Build the package with
```shell
python -m build
```
and install the produced wheel (in the dist directory).
Run training with
```shell
python -m palette train -c /path/to/config
```


## Acknowledgements
All credit for the implementation of the engine goes to Liangwei Jiang. That work is based on the following theoretical literature:
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Palette: Image-to-Image Diffusion Models](https://arxiv.org/pdf/2111.05826.pdf)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
and further makes use of the following projects:
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
- [LouisRouss/Diffusion-Based-Model-for-Colorization](https://github.com/LouisRouss/Diffusion-Based-Model-for-Colorization)
