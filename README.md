# edgeai-final

## Environment setup
- conda environment: installed packages are listed in `conda_env.yaml`.
- pip packages: required packages are listed in `requirements.txt`.

Note, you cannot actually install this way. Please use the steps below to install environment.
The requirements.txt and conda_env.yaml are for your reference only.

We use the miniconda distribution. Create the conda environment first:
```bash
conda create -n edgeai python=3.12
conda activate edgeai
```
Install pip dependencies
```bash
pip install -r requirements.txt
```

## Instructions on repoduction

After setting up the environment, use the below command to run the SLM inference.

```bash
CUDA_VISIBLE_DEVICES=0 python hqq_method/result_hqq.py
```