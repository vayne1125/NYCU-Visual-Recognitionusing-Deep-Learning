# NYCU Computer Vision 2025 Spring HW4
|StudentID|313551052|
|:-:|:-:|
|Name|王嘉羽 Chia-Yu, Wang|

## Introduction
This assignment focuses on image restoration, specifically removing rain and snow from degraded images to recover the original clean appearance. The designated model for this task is PromptIR, an all-in-one restoration model that automatically determines whether the input image is affected by rain or snow. 
Due to hardware limitations and the dataset images being 256x256 in size, I conducted most of my experiments using 128x128 images, aiming to achieve comparable results through various training strategies. Since the use of pretrained models was not allowed, one of my key strategies involved generating coarse pretrained weights using the provided dataset, followed by diverse data augmentation techniques to simulate different conditions and perform fine-tuning. Another notable strategy was patch testing: because the final output needs to be 256x256 while my model accepts 128x128 inputs, I divided each image into thirty-six 128x128 patches and used a Gaussian filter for weighted summation to reduce boundary artifacts. As a result, I achieved a PSNR of **31.10** on the public dataset, which I believe demonstrates the effectiveness of my approach.



## How to install
To install the necessary dependencies for this project, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/vayne1125/NYCU-Visual-Recognitionusing-Deep-Learning.git
cd HW4_Image-Restoration
```

### 2. (Optional) Create a virtual environment
It is recommended to use a virtual environment. You can use Anaconda or venv.
For Anaconda (Python 3.11.11):
```bash
conda create --name my_env python=3.11.11
conda activate my_env
```

### 3. Install the dependencies:
```bash
pip install -r requirements.txt
```
You also need to install [PyTorch](https://pytorch.org/). Choose the appropriate CUDA version based on your system. For CUDA 12.4:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```


### 4. Download the dataset and preprocess
- Download the dataset from [this link](https://drive.google.com/drive/folders/1Q4qLPMCKdjn-iGgXV_8wujDmvDpSI1ul)
- After downloading, extract the dataset into the `HW4_Image-Restoration` directory.
- Ensure the directory structure looks like this:
```
HW4_Image-Restoration
├── hw4_realse_dataset
│   ├── train
│   ├── test
├── datasets.py
├── requirement.txt
├── model.py
│   
.
.
.
```

### 5. Run the Code
- **For best results**, you can simply run the following commands without any additional changes to the configuration:
```bash
python train_part1.py
python train_part2.py
python test.py
```
Or, if you are a Windows user, you can directly use the following command:
```bash
./compile.bat
```
## Performance snapshot
Ranked 6th as of May 21.
<img src="./assets/snapshot.png">