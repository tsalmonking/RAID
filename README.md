# RAID
Source code for the paper "RAID: A Dataset for Testing the Adversarial
Robustness of AI-Generated Image Detectors"

# Setup

```
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

Set the dataset path once (all scripts read from this):
```
export TEST_PATH=/path/to/ELSA_TEST
```

# Detector Training

## Dataset Setup

First, create a directory `data` in the project root folder and download the [ELSA D3 dataset](https://huggingface.co/datasets/elsaEU/ELSA_D3) (or a subset):
```
python src/raid/scripts/elsa_downloader.py --split train --amount 200_000
python src/raid/scripts/elsa_downloader.py --split val
```

## Dataset Structure

The structure of the ELSA D3 dataset is as follows:
```
data
└── ELSA_D3					
      ├── train
            ├── real
            ├── sd_14
            ├── sd_21
            ├── sd_xl
            ├── df_if
      │── val   	
            │── real
            ├── sd_14
            ├── sd_21
            ├── sd_xl
            ├── df_if
            |      .
``` 

Then run the ```train_detectors.sh``` script

# Reproducing Results

All scripts are in `src/` and are run from the repo root:

## 1. Clean Baseline

```
src/run_clean_eval.sh
```

## 2. Adversarial Attacks

```
src/run_whitebox_attacks.sh          # white-box, eps 16/32
src/run_leave_one_out_attacks.sh     # leave-one-out ensemble, eps 16/32
src/run_full_ensemble_attacks.sh     # full 7-model ensemble, eps 8/16/32
```

## 3. Evaluation

```
src/run_evaluate_all.sh              # evaluate all detectors on all attack outputs
python3 src/build_results_table.py   # print paper tables (default: eps 16, 32)
python3 src/build_results_table.py 8 16 32
```

## 4. PGD Ablation

```
src/run_ablation_attacks.sh          # LOO sweep: step sizes x iterations x epsilons
src/run_ablation_noise.sh            # Gaussian + uniform noise baselines
```

Edit `DEVICE` and `HELD_OUT_DETECTORS` at the top of `run_ablation_attacks.sh` to split across GPUs.

# Evaluating and Attacking a Detector

## Running the attack
- `run_attack.sh`: Run the ensemble attack and evaluate the provided model on adversarial examples. Optionally saves adversarial examples for dataset creation.

```
  --generate                  saves the adversarial dataset at the provided output_dir        
  --eval_model                model to be evaluated
  --models                    model(s) to be attacked
  --device                    device(s) to run the models on                   
  --path_to_dataset           path to the dataset
  --dataset_type              subfolders | dataset | wang2020
  --output_dir                directory where the adversarial dataset is saved
  --epsilon                   perturbation budget (e.g. 16/255)
  --num_steps                 PGD iteration count
  --step_size                 PGD step size
  --ensembling_strategy       raw | avg | random
  --ensemble_loss             avg_ce
```

## Evaluating on the adversarial dataset
- `evaluate_detector.py`: Evaluate a model on a saved adversarial dataset.

```
  --model                     model to be evaluated
  --path_to_dataset           path to adv_dataset.pkl or subfolder dataset
  --dataset_type              dataset | subfolders
  --output_dir                directory where the evaluation results are saved
```

## Adding new detectors
[Evaluating and attacking a new detector](src/raid/models/example_model/model_wrapping.Md)

# File Structure

- `raid/attacks/` — ensemble attack, trackers, losses
- `raid/data/datasets.py` — dataloaders
- `raid/models/` — wrapped detectors
- `raid/attack_generate.py` — run adversarial attack and generate dataset
- `raid/evaluate_detector.py` — evaluate detector on a dataset
- `raid/generate_noise_baseline.py` — Gaussian/uniform noise baselines
- `external/` — third-party detector code
- `build_results_table.py` — aggregate results into paper tables
- `run_clean_eval.sh` — clean baseline evaluation
- `run_whitebox_attacks.sh` — white-box attacks
- `run_leave_one_out_attacks.sh` — leave-one-out ensemble attacks
- `run_full_ensemble_attacks.sh` — full 7-model ensemble attacks
- `run_evaluate_all.sh` — evaluate all detectors on all attack outputs
- `run_ablation_attacks.sh` — PGD hyperparameter ablation sweep
- `run_ablation_noise.sh` — noise baselines for ablation

# Detector Categorization

Detector | Detection Method | Architecture | Dataset | Preprocessing
--- | --- | --- | --- | ---
Ojha2023 | CLIP feature space + linear classifier | CLIP ViT-L/14 + linear | ForenSynths, DMs | CLIP normalize + CenterCrop 224
Corvi2023 | Modified ResNet50, no downsampling | ResNet50 | ProGAN, Latent Diffusion | ImageNet normalize
Cavia2024 | Patch-level scoring + global avg pooling | ResNet50, 1x1 convs | ForenSynths | ImageNet normalize + Resize 256
Chen2024 (ConvNeXt) | DRCT contrastive training | ConvNeXt | DRCT-2M | ImageNet normalize + CenterCrop 224
Chen2024 (CLIP) | DRCT contrastive training | CLIP ViT-L/14 | DRCT-2M | ImageNet normalize + CenterCrop 224
Koutlis2024 | CLIP intermediate blocks + projection | CLIP ViT-B/16 + linear | ForenSynths, Ojha, Tan | CLIP normalize + CenterCrop 224
Wang2020 | ResNet50 binary classifier | ResNet50 | ForenSynths | ImageNet normalize

# Reference Papers

Detector | Paper | Repository
--- | --- | ---
Ojha2023 | [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174) | [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
Corvi2023 | [On the detection of synthetic images generated by diffusion models](https://arxiv.org/abs/2211.00680) | [DMimageDetection](https://github.com/grip-unina/DMimageDetection)
Cavia2024 | [Real-Time Deepfake Detection in the Real-World](https://arxiv.org/abs/2406.09398) | [RealTime-DeepfakeDetection](https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld)
Chen2024 | [DRCT: Diffusion Reconstruction Contrastive Training](https://proceedings.mlr.press/v235/chen24ay.html) | [DRCT](https://github.com/beibuwandeluori/DRCT)
Koutlis2024 | [Leveraging Representations from Intermediate Encoder-blocks](https://arxiv.org/abs/2402.19091) | [rine](https://github.com/mever-team/rine)
Wang2020 | [CNN-generated images are surprisingly easy to spot...for now](https://arxiv.org/abs/1912.11035) | [CNNDetection](https://github.com/PeterWang512/CNNDetection)

# Licenses
The provided MIT License only applies to the `raid` directory. The code
contained in the `external` folder, provided by third-parties and modified in
some parts, has its own licenses that are included in each subfolder.
