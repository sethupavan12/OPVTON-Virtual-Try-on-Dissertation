# Thesis
This repo contains the RAW code I trained and tested the GAN model that I submitted for my bachelor thesis.
Now, more details about the project
# Abstract
Recent Image-based Virtual Try On (VTON) methods have gained widespread attention. The task involves generating a person image with an in-shop clothing item. Prior works have successfully addressed this problem including preserving clothing characteristics like logos, text, and patterns. However, occlusion remains a problem to be solved as existing methods produce artifacts when occlusion is present in the person image. This dissertation addresses the problem of occlusion in virtual try-on through proposed Occlusion-Preserving Virtual Try On Network (OP-VTON). At the core of OP-VTON pipeline are a geometric matching module that aligns the clothing item to the pose of the person in the input image and a powerful image generator that takes aligned clothing item along with other guiding information to generate a convincing try-on result. OP-VTON is evaluated qualitatively and quantitatively on VITON dataset and then compared with state-of-the-art works from the virtual try-on domain. The qualitative evaluation revealed that OP-VTON outperforms state-of-the-art models in occlusion scenarios, while the quantitative evaluation demonstrated that OP-VTON outperforms state-of-the-art models in terms of FID and SSIM scores. Additionally, this dissertation also trains a latent diffusion model on VITON and DeepFashion datasets through use of a recent pioneering work called Control Net to enable novel text-guided virtual try-on image generation capabilities without losing the input image identity.


# Architecture
![image](https://github.com/sethupavan12/OPVTON-Virtual-Try-on-Dissertation/assets/60856766/879ee57f-7de3-45a1-86b8-a136ec2d7fa3)


# Qualitative Comparision of OPVTON and other SOTA models
![image](https://github.com/sethupavan12/OPVTON-Virtual-Try-on-Dissertation/assets/60856766/ed26ff4c-e0c6-4d7e-8157-d6ac3a71229f)


# Quantitative Comparision of OPVTON and other SOTA models
![image](https://github.com/sethupavan12/OPVTON-Virtual-Try-on-Dissertation/assets/60856766/a618db8e-85ea-4fdc-ba4c-b30917d5ebb3)


## Maintaining
I want to make this repo as clean as possible. Given that this is a bit of a niche field, I have other ideas but sadly no time to make it happen.
Will clean the code base and add sensible instructions later.

note: I leftout the controlnet code part from this repo 
