# Generative Adversarial Networks in PyTorch

In this repository you will find PyTorch implementations of popular GAN frameworks, run on interesting datasets (such as celebrity faces, anime faces, etc)

### Requirements
Python3.6

### Installation
```pip install -r requirements.txt```

### Models implemented
- [DCGAN](#dcgan)
- [BIGAN / Adverversarially Learned Inference](#adversarially-learned-inference)
----------------------------------
# [DCGAN](https://arxiv.org/abs/1511.06434)

| Dataset (Res) | Generated Samples  | Interpolations |
| ------------- | ------------- | ------------- |
| [Anime](https://github.com/Mckinsey666/Anime-Face-Dataset)  (64 x 64)  | ![DCGAN Anime](/imgs/anime_small_dcgan.png)  | ![DCGAN Anime Interpolation](/imgs/anime_small_interp_dcgan.png)  |
| [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (64 x 64)  | ![DCGAN CelebA](/imgs/celeba_small_dcgan.png)  | ![DCGAN CelebA Interpolation](/imgs/celeba_small_interp_dcgan.png)  |

# [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704)
| Dataset (Res) | Generated Samples  | Original -> Reconstructions |
| ------------- | ------------- | ------------- |
| [Anime](https://github.com/Mckinsey666/Anime-Face-Dataset)  (64 x 64)  | ![ALI Anime](/imgs/anime_small_ali.png)  | ![ALI Anime Interpolation](/imgs/anime_small_recons_ali.png)  |

More implementations coming soon. Stay tuned!
