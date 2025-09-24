# RoPECraft: Training-Free Motion Transfer with Trajectory-Guided RoPE Optimization on Diffusion Transformers

## Accepted to NeurIPS 2025

[\*Ahmet Berke Gokmen](https://berkegokmen1.github.io/)<sup>1,2</sup>, [\*Yiğit Ekin](https://yigitekin.github.io/)<sup>1</sup>, [\*Bahri Batuhan Bilecen](https://three-bee.github.io)<sup>1,3,4</sup>, and [Aysegul Dundar](http://www.cs.bilkent.edu.tr/~adundar/)<sup>1</sup>

<sup>1</sup>Bilkent University, <sup>2</sup>INSAIT, Sofia University “St. Kliment Ohridski”, <sup>3</sup>ETH Zurich, <sup>4</sup>Max Planck Institute

\* Equal Contribution

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Project Website](https://img.shields.io/badge/Project_website-red.svg)](https://berkegokmen1.github.io/RoPECraft) [![arXiv](https://img.shields.io/badge/arXiv-2505.13344-b31b1b.svg)](https://arxiv.org/abs/2505.13344)

---

We propose RoPECraft, a training-free video motion transfer method for diffusion transformers that operates solely by modifying their rotary positional embeddings (RoPE). We first extract dense optical flow from a reference video, and utilize the resulting motion offsets to warp the complex-exponential tensors of RoPE, effectively encoding motion into the generation process. These embeddings are then further optimized during denoising time steps via trajectory alignment between the predicted and target velocities using a flow-matching objective. To keep the output faithful to the text prompt and prevent duplicate generations, we incorporate a regularization term based on the phase components of the reference video’s Fourier transform, projecting the phase angles onto a smooth manifold to suppress high-frequency artifacts. Experiments on benchmarks reveal that RoPECraft outperforms all recently published methods, both qualitatively and quantitatively.

## Setup

The code has been tested using `Debian GNU/Linux 13 (trixie)` and single `NVIDIA H200` gpu.

Recommended CUDA Version: `12.4.1`<br>
Recommended GCC Version: `12.4.0`

```
conda create -n ropecraft python=3.11 -y
conda activate ropecraft
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Dataset

We've mainly utilized [DAVIS](https://davischallenge.org/) as our reference motion video dataset. However our method generalizes to any other video.
For sample prompts, please refer to our prompts dataset at this [link](https://drive.google.com/drive/folders/1xWLPBKTzsEifiDVBbllJZ9tH790L9Frh).

## Running the Demo
We've left some sample videos from DAVIS dataset for you to try our demo. 

Simply run the following command on a machine with GPU VRAM > 40GB. It'll automatically download the required checkpoints (only once) from Huggingface.

```bash
python generate.py \
    --input_video assets/blackswan.mp4 \
    --prompt "A child in a duck costume paddles a small inflatable boat across a pond." \
    --output_dir output \
    --start_with_uv_warped 1 \
    --frames 49 \
    --num_optim_steps 5 \
    --n_replace_gt_mod 10
```

or

```bash
python generate.py \
    --input_video assets/bus.mp4 \
    --prompt "A herd of sheep marches along the street while a shepherd watches from the sidewalk." \
    --output_dir output \
    --start_with_uv_warped 1 \
    --frames 49 \
    --num_optim_steps 5 \
    --n_replace_gt_mod 10
```

## Evaluation

The paper details a new motion transfer metric called `Fréchet Trajectory Distance` (FTD). You can compute a sample FTD metric from a generated video provided in `assets/`.

```bash
python ftd.py \
    --reference_video assets/blackswan.mp4 \
    --target_video assets/output/A_child_in_a_duck_co.mp4 \
    --mask_path assets/mask/blackswan \
    --num_points 100
```

---

If you found our work useful, please don't forget to cite our work.

```
@misc{gokmen2025ropecrafttrainingfreemotiontransfer,
      title={RoPECraft: Training-Free Motion Transfer with Trajectory-Guided RoPE Optimization on Diffusion Transformers}, 
      author={Ahmet Berke Gokmen and Yigit Ekin and Bahri Batuhan Bilecen and Aysegul Dundar},
      year={2025},
      eprint={2505.13344},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.13344}, 
}
```


Copyright 2025 [Bilkent DLR](https://dlr.bilkent.edu.tr/). Licensed under the Apache License, Version 2.0 (the "License").
