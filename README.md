# RoPECraft: Training-Free Motion Transfer with Trajectory-Guided RoPE Optimization on Diffusion Transformers

## Accepted to NeurIPS 2025

[\*Ahmet Berke Gokmen](https://berkegokmen1.github.io/)<sup>1,2</sup>, [\*Yiğit Ekin](https://yigitekin.github.io/)<sup>1</sup>, [\*Bahri Batuhan Bilecen](https://three-bee.github.io)<sup>1,3,4</sup>, and [Aysegul Dundar](http://www.cs.bilkent.edu.tr/~adundar/)<sup>1</sup>

<sup>1</sup>Bilkent University, <sup>2</sup>INSAIT, Sofia University “St. Kliment Ohridski”, <sup>3</sup>ETH Zurich, <sup>4</sup>Max Planck Institute

\* Equal Contribution

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Project Website](https://img.shields.io/badge/Project_website-red.svg)](https://berkegokmen1.github.io/RoPECraft) [![arXiv](https://img.shields.io/badge/arXiv-2505.13344-b31b1b.svg)](https://arxiv.org/abs/2505.13344)

---
CODE WILL BE RELEASED BEFORE THE END OF SEPTEMBER 2025.
***

We propose RoPECraft, a training-free video motion transfer method for diffusion transformers that operates solely by modifying their rotary positional embeddings (RoPE). We first extract dense optical flow from a reference video, and utilize the resulting motion offsets to warp the complex-exponential tensors of RoPE, effectively encoding motion into the generation process. These embeddings are then further optimized during denoising time steps via trajectory alignment between the predicted and target velocities using a flow-matching objective. To keep the output faithful to the text prompt and prevent duplicate generations, we incorporate a regularization term based on the phase components of the reference video’s Fourier transform, projecting the phase angles onto a smooth manifold to suppress high-frequency artifacts. Experiments on benchmarks reveal that RoPECraft outperforms all recently published methods, both qualitatively and quantitatively.

Copyright 2025 [Bilkent DLR](https://dlr.bilkent.edu.tr/). Licensed under the Apache License, Version 2.0 (the "License").
