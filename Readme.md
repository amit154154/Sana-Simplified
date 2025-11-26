# Sana / Sana Sprint / Sana Video – Research Experiments

This repo is a **research playground** for experiments on top of `SANA` and `Sana Sprint` diffusion models.  
It currently includes **prototype code**, **one-off experiments**, and **work-in-progress training scripts** (LoRA, DreamBooth-style tuning, zoom conditioning, control net etc).



## Sana-Video Tom & Jerry Experiment

This experiment fine-tunes **SANA-Video (2B)** with **LoRA** on a dataset of _Tom and Jerry_ clips resized to **224×224**.  
The base model is originally trained for higher resolutions (e.g. 480p), so its **zero-shot** generations at 224×224 are:

-- off-style compared to classic 2D slapstick cartoons  
-- less consistent in line art, color palette, and motion
 
---

### Training Setup

#### Objective
[![Hugging Face – sanavideo-tomjerry-lora-r16-v1](https://img.shields.io/badge/HuggingFace-sanavideo--tomjerry--lora--r16--v1-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/AmitIsraeli/sanavideo-tomjerry-lora-r16-v1)  
[![Weights & Biases – Tom & Jerry Runs](https://img.shields.io/badge/Weights_&_Biases-Tom_%26_Jerry%20Runs-2c8ebb?style=for-the-badge&logo=weightsandbiases&logoColor=white)](https://wandb.ai/amit154154/sana-video-tomjerry)

- Start from **base SANA-Video 2B**  
- Freeze everything except **LoRA adapters on all linear layers in the diffusion transforme**  
- Train with a **single class prompt** describing the Tom & Jerry world  
- Resolution: **224×224**  
- Dataset: curated Tom & Jerry-style clips

#### Key Hyperparameters

| Component          | Value                                |
|--------------------|--------------------------------------|
| Base model         | `SANA-Video_2B_480p_diffusers`       |
| Resolution         | 224 × 224                            |
| Batch size         | 8                                    |
| Optimizer          | AdamW / AdamW8bit (LoRA params only) |
| Learning rate      | 2e-4                                 |
| LoRA rank          | 16                                   |
| LoRA alpha         | 32                                   |
| LoRA dropout       | 0.1                                  |
| Training objective | Flow Matching (velocity prediction)  |
| Steps              | 10,000                               |
| clip length        | 81 frames                            |



---

### Class Prompt (Style Anchor)

The first training run used a **single class prompt** to define the style:

```text
A vintage slapstick 2D cartoon scene of a grey cat chasing a small brown mouse in a colorful house, Tom and Jerry style, bold outlines, limited color palette, exaggerated expressions, smooth character motion.
```

---

### Training Progression (Class LoRA)

During training, videos were generated every few steps with a fixed seed and CFG to visualize how the LoRA gradually pulls SANA-Video toward the Tom & Jerry domain.

All of these samples live in:

- `assets/tom_and_jerry_all_assets/tomjerry_video_class_lora16_seed69420_cfg4`

#### Checkpoint Comparison (Same Seed, Same Prompt)

| Step   | Video |
|--------|-------|
| Base   | ![Base](assets/tom_and_jerry_all_assets/tomjerry_video_class_lora16_seed69420_cfg4/tomjerry_base_seed69420.gif) |
| 100    | ![Step 100](assets/tom_and_jerry_all_assets/tomjerry_video_class_lora16_seed69420_cfg4/tomjerry_lora_step000100_seed69420.gif) |
| 1,000  | ![Step 1k](assets/tom_and_jerry_all_assets/tomjerry_video_class_lora16_seed69420_cfg4/tomjerry_lora_step001000_seed69420.gif) |
| 2,000  | ![Step 2k](assets/tom_and_jerry_all_assets/tomjerry_video_class_lora16_seed69420_cfg4/tomjerry_lora_step002000_seed69420.gif) |
| 5,000  | ![Step 5k](assets/tom_and_jerry_all_assets/tomjerry_video_class_lora16_seed69420_cfg4/tomjerry_lora_step005000_seed69420.gif) |
| 7,500  | ![Step 7.5k](assets/tom_and_jerry_all_assets/tomjerry_video_class_lora16_seed69420_cfg4/tomjerry_lora_step007500_seed69420.gif) |
| 10,000 | ![Step 10k](assets/tom_and_jerry_all_assets/tomjerry_video_class_lora16_seed69420_cfg4/tomjerry_lora_step010000_seed69420.gif) |

when all examples are from seed 69420 and the same class prompt  

---

### Text-Conditioned Experiments

After the class-LoRA training, the next step was to test how much **text conditioning ability** remained:

- Same LoRA, **different prompts** for different scenarios.
- Goal: see whether the model can respond to new text instructions **without losing** the learned Tom & Jerry style.

All text-conditioned videos are collected under:


#### Scenarios & Prompts

| Full prompt | Summary to prompt | Video |
|-------------|-------------------|-------|
| A vintage slapstick 2D cartoon scene of a grey cat chasing a small brown mouse in a colorful house, Tom and Jerry style, bold outlines, limited color palette, exaggerated expressions, smooth character motion. | Class baseline (same style anchor as training prompt) | ![Class baseline](assets/tom_and_jerry_all_assets/fine_tuned_class_textcond/tomjerry_classcond_seed32420.gif) |
| A classic slapstick 2D cartoon scene of a grey cat sneaking after a small brown mouse in a cozy living room at night, Tom and Jerry style, bold outlines, warm lamp lighting, deep shadows, exaggerated sneaking poses, smooth tiptoe animation. | Night living room sneaking (tests lighting and sneaking motion) | ![At night](assets/tom_and_jerry_all_assets/fine_tuned_class_textcond/tomjerry_at_night_seed32420.gif) |
| A retro slapstick 2D cartoon scene of a grey cat and small brown mouse chasing each other up and down a staircase in a big old mansion, Tom and Jerry style, bold outlines, muted vintage colors, dust clouds, stretched limbs, smooth looping motion. | Vertical staircase chase in an old mansion | ![Vertical chase](assets/tom_and_jerry_all_assets/fine_tuned_class_textcond/tomjerry_vertical_chase_oldbuilding_32420.gif) |
| A vintage slapstick 2D cartoon scene of a grey cat chasing a small brown mouse in a colorful house, Tom and Jerry style, bold outlines, limited color palette, exaggerated expressions, smooth character motion. tom is eating a pizza. | Tom eating pizza indoors (object + action added) | ![Tom eating pizza](assets/tom_and_jerry_all_assets/fine_tuned_class_textcond/tomjerry_tom_eatingpizza_32420.gif) |
| A vintage slapstick 2D cartoon scene of a grey cat being outsmarted by a small brown mouse in a sunny backyard, Tom and Jerry style, bold outlines, vibrant green grass and blue sky, exaggerated pranks, falling objects, smooth expressive animation. | Prank outside in a sunny backyard | ![Prank outside](assets/tom_and_jerry_all_assets/fine_tuned_class_textcond/tomjerry_prank_outside_32420.gif) |

### Metrics & Evaluation (Template)

TBD


### Next Steps:

1.	Characterize training stability: Run a 15k-step LoRA-64 experiment (LR = 1e-4 with warmup) to study the early-step loss explosion and compare dynamics against the original LoRA-16 run.
2.	VLM-driven control & labeling: Use a vision–language model to auto-label clips with richer descriptions and control tags, and design a prompt paradigm for video → text → video conditioning.
3.	Model compression & sampling study: Systematically evaluate distillation, pruning, and different sampling-step budgets, measuring their impact on quality and downstream metrics.