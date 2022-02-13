# HI2I: Hypercomplex Image-to-Image Translation
Official PyTorch repository for Hypercomplex Image-to-Image Transaltion

Eleonora Grassucci, Luigi Sigillo, Aurelio Uncini, and Danilo Comminiello

:warning: This repository is under construction  ⚠️


### Abstract

Image-to-image translation (I2I) aims at transferring the content representation from an input domain to an output one, bouncing along different target domains. Recent I2I generative models which gain outstanding results in this task comprise a set of diverse deep networks each with tens of million parameters. Moreover, images are usually three-dimensional being composed of RGB channels and common neural models do not take dimensions correlation into account, losing beneficial information. In this paper, we propose to leverage hypercomplex algebra properties to define lightweight I2I generative models capable of preserving pre-existing relations among images dimensions, thus exploiting additional input information. On manifold I2I benchmarks, we show how the proposed Quaternion StarGANv2 and parameterized hypercomplex StarGANv2 (PHStarGANv2) reduce parameters and storage memory amount while ensuring high domain translation performance and good image quality as measured by FID and LPIPS score.
