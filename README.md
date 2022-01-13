# Max-Margin Contrastive Learning

This is a pytorch implementation for the paper Max-Margin Contrastive Learning accepted to AAAI 2022.

This repository is based on [SimCLR-pytorch](https://github.com/AndrewAtanov/simclr-pytorch). 

## Set-up environment
- `conda env create -f mmcl_env.yaml`
- `conda activate mmcl`
  
## Prepare data
- `export IMAGENET_PATH=path_to_dataset`
- Find ImageNet-100 classes [here](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt)

## Train 
- Train MMCL models using one of the following commands:
- ImageNet-1k:
  - `python train.py --config configs/imagenet1k_mmcl_pgd.yaml`
  - `python train.py --config configs/imagenet1k_mmcl_inv.yaml`
- ImageNet-100:
  - `python train.py --config configs/imagenet100_mmcl_pgd.yaml`
  - `python train.py --config configs/imagenet100_mmcl_inv.yaml`

## Linear-Evaluation
- ImageNet-1k models:
  - `python train.py --config configs/imagenet_eval.yaml --encoder_ckpt path_to_experiment_folder//checkpoint-500400.pth.tar`
- ImageNet-100 models:
  - `python train.py --config configs/imagenet100_eval.yaml --encoder_ckpt path_to_experiment_folder/checkpoint-198000.pth.tar`
- Experiment folder can be found in `logs/exman-train.py/runs/X`
- Following are some results and pretrained models. 

| Model | Linear-Evaluation | Pretrained Model |
| --- | --------- | -------- |
| ImageNet-1K MMCL PGD | 63.7 | [here](http://www.cis.jhu.edu/~ashah/MMCL/imagenet1k_mmcl_pgd.pth.tar)
| ImageNet-100 MMCL PGD | 80.7 | [here](http://www.cis.jhu.edu/~ashah/MMCL/imagenet100_mmcl_pgd.pth.tar)

## Citation
If you find this repository useful in your research, please cite:
```
@misc{shah2021maxmargin,
      title={Max-Margin Contrastive Learning}, 
      author={Anshul Shah and Suvrit Sra and Rama Chellappa and Anoop Cherian},
      year={2021},
      eprint={2112.11450},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```