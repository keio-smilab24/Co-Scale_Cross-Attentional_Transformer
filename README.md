# Co-Scale Cross-Attentional Transformer for Rearrangement Target Detection

[[paper](https://arxiv.org/abs/2407.05063)]

Haruka Matsuo, Shintaro Ishikawa and Komei Sugiura

Rearranging objects (e.g. vase, door) back in their original positions is one
of the most fundamental skills for domestic service robots (DSRs). In
rearrangement tasks, it is crucial to detect the objects that need to be
rearranged according to the goal and current states. In this study, we focus on
Rearrangement Target Detection (RTD), where the model generates a change mask
for objects that should be rearranged. Although many studies have been
conducted in the field of Scene Change Detection (SCD), most SCD methods often
fail to segment objects with complex shapes and fail to detect the change in
the angle of objects that can be opened or closed. In this study, we propose a
Co-Scale Cross-Attentional Transformer for RTD. We introduce the Serial Encoder
which consists of a sequence of serial blocks and the Cross-Attentional Encoder
which models the relationship between the goal and current states. We built a
new dataset consisting of RGB images and change masks regarding the goal and
current states. We validated our method on the dataset and the results
demonstrated that our method outperformed baseline methods on $F_1$-score and
mean IoU.


## Setup
```bash
git clone https://github.com/keio-smilab24/Co-Scale_Cross-Attentional_Transformer
cd Co-Scale_Cross-Attentional_Transformer
```

We assume the followiing environment for our experiments:
* Python 3.6.8
* PyTorch version 1.8.0 with CUDA 11.1 support

```bash
pyenv virtualenv 3.6.8 cscat
pyenv local cscat

# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.6.0.66 wandb==0.15.5 tqdm scikit-learn timm einops
```

Follow the instructions in the [sscdnet](https://github.com/kensakurada/sscdnet/tree/master) (["Environments"](https://github.com/kensakurada/sscdnet/tree/master?tab=readme-ov-file#environments)) and build GCC and correlation layer package.

```bash
git clone https://github.com/kensakurada/sscdnet.git
cd sscdnet
# build GCC and correlation layer package
```
```bash
cd correlation_package
# Modify setup.py to suit your environment
python setup.py build
```


## Dataset
Download our dataset from [here](https://drive.google.com/file/d/109LUdkgxYV2XpbPN9Dh-7Vj0-ZR1kkvr/view?usp=drive_link).
We expect the directory structure to be the following:

```bash
./data
└── RTD_dataset
　   ├── train
　   │    ├── goal         # *.png
　   │    ├── current      # *.png
　   │    └── mask         # *.png
　   ├── valid
　   ...
　   └── test
　   　   ├── goal
　   　   ├── current
　   　   └── mask
```
We used [AI2-THOR](https://github.com/allenai/ai2thor) and referenced the [2022 AI2-THOR Rearrangement Challenge](https://github.com/allenai/ai2thor-rearrangement/tree/2022-challenge-v0) to build this dataset.


## Train
```bash
python train.py train --datadir data/RTD_dataset --checkpointdir log --num-workers 4
```

## Evaluation
```bash
python test.py test --datadir data/RTD_dataset --checkpointdir log --model pretrained_model.pth --image
```

FYI: Model checkpoint is available [here](https://drive.google.com/file/d/1QdN1StaYx6lPFOnSsxsb_08bwrqKMdB9/view?usp=drive_link).