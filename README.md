# HI2I: Hypercomplex Image-to-Image Translation
Official PyTorch repository for Hypercomplex Image-to-Image Transaltion

Eleonora Grassucci, Luigi Sigillo, Aurelio Uncini, and Danilo Comminiello

:warning: This repository is under construction  ⚠️


### Abstract

Image-to-image translation (I2I) aims at transferring the content representation from an input domain to an output one, bouncing along different target domains. Recent I2I generative models which gain outstanding results in this task comprise a set of diverse deep networks each with tens of million parameters. Moreover, images are usually three-dimensional being composed of RGB channels and common neural models do not take dimensions correlation into account, losing beneficial information. In this paper, we propose to leverage hypercomplex algebra properties to define lightweight I2I generative models capable of preserving pre-existing relations among images dimensions, thus exploiting additional input information. On manifold I2I benchmarks, we show how the proposed Quaternion StarGANv2 and parameterized hypercomplex StarGANv2 (PHStarGANv2) reduce parameters and storage memory amount while ensuring high domain translation performance and good image quality as measured by FID and LPIPS score.

### Model Architecture (from [StarGANv2](https://github.com/clovaai/stargan-v2))
<img src="PHStarGANv2_arch.png">









### Cite

Please cite our work if you found it useful:

```
@article{grassucci2022HI2I,
      title={Hypercomplex Image-to-Image Translation}, 
      author={Grassucci, Eleonora and Sigillo, Luigi and Uncini, Aurelio and Comminiello, Danilo},
      year={2022},
      journal={Under review}
}
```

#### Interested in Quaternion and Hypercomplex Generative Models?

Check also: 

* Lightweight Convolutional Neural Network by Hypercomplex Parameterization, _Under Review_, 2021 [[Paper](https://arxiv.org/pdf/2110.04176.pdf)] [[GitHub](https://github.com/elegan23/hypernets)].
* Quaternion-Valued Variational Autoencoder, _ICASSP_, 2021 [[Paper](https://arxiv.org/pdf/2010.11647.pdf)] [[GitHub](https://github.com/eleGAN23/QVAE)].
* An Information-Theoretic Perspective on Proper Quaternion Variational Autoencoders, _Entropy_, 2021 [[Paper](https://www.mdpi.com/1099-4300/23/7/856)] [[GitHub](https://github.com/eleGAN23/QVAE)].
* Quaternion Generative Adversarial Networks, _Generative Adversarial Learning: Architectures and Applications, editors: Dr Roozbeh Razavi-Far, Dr Ariel Ruiz-Garcia, Professor Vasile Palade, Professor Jürgen Schmidhuber, Springer_, Jan 2022. [[Paper](https://arxiv.org/pdf/2104.09630.pdf)][[GitHub](https://github.com/eleGAN23/QGAN)].
