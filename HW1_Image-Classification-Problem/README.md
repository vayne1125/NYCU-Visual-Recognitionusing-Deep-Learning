# NYCU Computer Vision 2025 Spring HW1
|StudentID|313551052|
|:-:|:-:|
|Name|王嘉羽 Chia-Yu, Wang|

## Introduction

## How to install
To install the necessary dependencies for this project, follow these steps:

1. **Clone the repository**
```bash
git clone https://github.com/vayne1125/NYCU-Visual-Recognitionusing-Deep-Learning.git
cd HW1_Image-Classification-Problem
```

2. **(Optional) Create a virtual environment**
It is recommended to use a virtual environment. You can use Anaconda or venv.
For Anaconda (Python 3.11.11):
```bash
conda create --name my_env python=3.11.11
conda activate my_env
```

3. **Install the dependencies:**
```bash
pip install -r requirements.txt
```
You also need to install [PyTorch](https://pytorch.org/). Choose the appropriate CUDA version based on your system. For CUDA 12.4:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```


4. **Download the dataset**
- Download the dataset from [this link](https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view)
- After downloading, extract the dataset into the `HW1_Image-Classification-Problem` directory.
- Ensure the directory structure looks like this:
```
HW1_Image-Classification-Problem
├── data
│   ├── test
│   ├── train
│   └── val
├── datasets
│   └── datasets.py
├── requirement.txt
├── params
│   └── best_params.pt
.
.
.
```


## Performance snapshot
