# Contrapositive Local Inference

This repository contains the code and resources for the *contrapositive inference* model presented in the following paper:  

Omid Kashefi, Rebecca Hwa, 2021. "[Contrapositive Local Class Inference.](https://aclanthology.org/2021.wnut-1.41.pdf)" in *EMNLP Workshop on Noisy User-generated Text (W-NUT)*.


The model is implemented as a `transfer function` on top of `ctrl-gen` model [(Hu et al., 2017)](https://arxiv.org/pdf/1703.00955.pdf), modified with an additional `conciseness loss` constraint to ensure the class transference with minimal changes to the original greater-context.

## Usage

``` bash
python3 model\contrapositive-tf.py --config config.[task]
```
