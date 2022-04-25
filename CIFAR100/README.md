### Code for MMCL_INV and MMCL_PGD on CIFAR-100

#### Configuration
- Python 3.7
- numpy==1.20.0
- pandas==1.1.5
- opencv_python==4.5.1.48
- torch==1.3.1
- torchvision==0.4.2
- tqdm==4.56.0
- termcolor==1.1.0
- matplotlib==3.4.1
- Pillow==8.2.0
- scikit_learn==0.24.2

#### Scripts
- `main.py` : Main training script
- `svm_losses.py` : Includes our criterion for MMCL_INV and MMCL_PGD
- `linear.py` : For training a linear layer on top of a frozen backbone and evaluating it
- `utils.py` : utilities for training, dataset loading and transforms
- `solvers.py` : PGD solver implementations
- `model.py` : resnet model definition

#### Example scripts
- Example MMCL_INV script `python main.py --criterion_to_use=mmcl_inv --run_name='X'` 
- Example MMCL_PGD script `python main.py --criterion_to_use=mmcl_pgd --run_name='X'`
- For evaluation of a trained model, run `python linear.py --model_path ../results/cifar100/X/model_400.pth`

#### Acknowledgements
- The code is modified from the [HCL](https://github.com/joshr17/HCL/tree/main/image) repository. 