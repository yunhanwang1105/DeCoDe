<div align="center">

<h1>Decompose, Compare, and Decide:<br>Multimodal LLMs are Implicit Few-Shot Learners</h1>

<p>
  <a href="https://yunhanwang1105.github.io/">Yunhan Wang</a> ·
  <a href="https://esh04.github.io/">Eshika Khandelwal</a> ·
  <a href="https://edsonroteia.github.io/">Edson Araujo</a> ·
  <a href="https://walidbousselham.com/">Walid Bousselham</a> ·
  <a href="https://ninatu.github.io/">Nina Shvetsova</a> ·
  <a href="https://hildekuehne.github.io/">Hilde Kuehne</a>
</p>

<p><em>Tübingen AI Center, University of Tübingen, Germany</em></p>

<p>Official repository for the DeCoDe paper at ECCV 2026.</p>

<p>
  <a href="https://yunhanwang1105.github.io/publications/DeCoDe/"><img src="https://img.shields.io/badge/Project_Page-Page-64fefe" alt="Project Page"></a>
  &emsp;
  <a href="https://arxiv.org/abs/2607.00125"><img src="https://img.shields.io/badge/arXiv-PDF-a92225" alt="arXiv"></a>
</p>

</div>

## Repository structure

```text
data/              # dataset roots or symlinks
src/               # DeCoDe source code
episode_logs/      # fixed few-shot episode sampling logs and runtime outputs
cache/             # default model cache, e.g. Qwen3-VL-8B-Instruct
configs/           # YAML experiment configs
```

## Environment setup

Create a conda environment and install PyTorch 2.10:

```bash
conda create -n decode python=3.11 -y
conda activate decode

pip install "torch==2.10.*"
pip install -r requirements.txt
```

If your CUDA setup needs a specific PyTorch wheel, install the matching PyTorch 2.10 build first, then install the remaining requirements.

## Data

Download or symlink datasets under `data/` so they match the expected reader layout. See [data/README.md](data/README.md) for the dataset-specific folder structure and download notes.

## Running Qwen3-VL DeCoDe evaluation

After setting up the environment and data, run:

```bash
python src/scripts/run_few_shot.py --config configs/qwen3vl_decompose.yaml --dataset mini
```

The default config uses `Qwen/Qwen3-VL-8B-Instruct`, `decompose_semantic` support/query prompting, and writes prediction JSONL files to `episode_logs/runs/<dataset>/`.
Pairwise support/query comparisons are evaluated in batches controlled by `experiment.batch_size` or `--batch_size`.

Prompt modes:

- `decompose_semantic`: compare support/query images with the support class label in the prompt.
- `decompose_anonymous`: compare support/query images without class labels.
Add `--domain_info` to either prompt mode to use dataset-specific domain wording, such as `aircraft variant`, `bird species`, or `action category`.

Configured datasets: `mini`, `cub`, `dogs`, `arabicsign`, `yoga`, `hieroglyph`, `ucf`, `aircraft`, `domain`, `lego`, `butterfly`, and `industrial`.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026decomposecomparedecidemultimodal,
  title         = {Decompose, Compare, and Decide: Multimodal LLMs are Implicit Few-Shot Learners},
  author        = {Yunhan Wang and Eshika Khandelwal and Edson Araujo and Walid Bousselham and Nina Shvetsova and Hilde Kuehne},
  year          = {2026},
  eprint        = {2607.00125},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.00125}
}
```
