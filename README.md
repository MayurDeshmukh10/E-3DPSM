# E-3DPSM: A State Machine for Event-based Egocentric 3D Human Pose Estimation [CVPR'24]
<center>

<!-- Christen Millerdurai<sup>1,2</sup>, Hiroyasu Akada<sup>1</sup>, Jian Wang<sup>1</sup>, Diogo Luvizon<sup>1</sup>, Christian Theobalt<sup>1</sup>, Vladislav Golyanik<sup>1</sup> -->

<sup>1</sup> Max Planck Institute for Informatics, SIC  &nbsp; &nbsp; &nbsp; &nbsp; <sup>2</sup> Saarland University, SIC  

</center>

## Official PyTorch implementation

[Project page](https://4dqv.mpi-inf.mpg.de/EventEgo3D/) | [Paper](https://arxiv.org/abs/2404.08640) 

<p align="center">
<img src="images/teaser.gif" alt="EventEgo3D" height="172"  /></br>
</p>

### Abstract

Event cameras offer multiple advantages in monocular egocentric 3D human pose estimation from head-mounted devices, such as millisecond temporal resolution, high dynamic range, and negligible motion blur. Existing methods effectively leverage these properties, but suffer from low 3D estimation accuracy, insufficient in many applications (e.g., immersive VR/AR). This is due to the design not being fully tailored towards event streams (e.g., their asynchronous and continuous nature), leading to high sensitivity to self-occlusions and temporal jitter in the estimates. This paper rethinks the setting and introduces E-3DPSM, an event-driven continuous pose state machine for event-based egocentric 3D human pose estimation. E-3DPSM aligns continuous human motion with fine-grained event dynamics; it evolves latent states and predicts continuous changes in 3D joint positions associated with observed events, which are fused with direct 3D human pose predictions, leading to stable and drift-free final 3D pose reconstructions. E-3DPSM runs in real-time at $80$ Hz on a single workstation and sets a new state of the art in experiments on two benchmarks, improving accuracy by up to $19\%$ (MPJPE) and temporal stability by up to $64\%$.

### Advantages of Event Based Vision
High Speed Motion                      |  Low Light Performance          
:-------------------------:|:-------------------------:|
| <img src="images/fast_motion.gif" alt="High Speed Motion" width="350"/> | <img src="images/low_light.gif" alt="Low Light Performance" width="350"/> |

### EventEgo3D

<p align="center">
<img src="images/method_figure.png" alt="E-3DPSM" /></br>
</p>

## Usage
-----
- [EventEgo3D: 3D Human Motion Capture from Egocentric Event Streams \[CVPR'24\]](#eventego3d-3d-human-motion-capture-from-egocentric-event-streams-cvpr24)
  - [Official PyTorch implementation](#official-pytorch-implementation)
    - [Abstract](#abstract)
    - [Advantages of Event Based Vision](#advantages-of-event-based-vision)
    - [EventEgo3D](#eventego3d)
  - [Usage](#usage)
    - [Installation](#installation)
      - [Dependencies](#dependencies)
      - [Pretrained Model](#pretrained-model)
    - [Datasets](#datasets)
    - [Training](#training)
    - [Evaluation](#evaluation)
      - [EE3D-S](#ee3d-s)
      - [EE3D-R](#ee3d-r)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
------

### Installation

Clone the repository
```bash
git clone https://github.com/Chris10M/EventEgo3D.git
cd EventEgo3D
```

#### Dependencies
Create a conda enviroment from the file 
```bash
conda env create -f EventEgo3D.yml
```
Next, install  **[ocam_python](https://github.com/Chris10M/ocam_python.git)** using pip
```bash
pip3 install git+https://github.com/Chris10M/ocam_python.git
```


#### Pretrained Model 

The pretrained model ```best_model_state_dict.pth``` can be found [here](https://eventego3d.mpi-inf.mpg.de/CVPR/best_model_state_dict.pth). Please place the model in the following folder structure.

```bash
EventEgo3D
|
└── saved_models
         |
         └── best_model_state_dict.pth
```


### Datasets

The datasets can obtained by executing the files in [`dataset_scripts`](./dataset_scripts/). For detailed information, refer [here](./dataset_scripts/). 


### Training

For training, ensure [EE3D-S](./dataset_scripts#ee3d-s), [EE3D-R](./dataset_scripts#ee3d-r) and [EE3D[BG-AUG]](./dataset_scripts#ee3d-bg-aug) are present. 
The batch size and checkpoint path can be specified with the following environment variables, ```BATCH_SIZE``` and ```CHECKPOINT_PATH```.

```bash
python train.py 
```

### Evaluation

#### EE3D-S 
For evaluation, ensure [EE3D-S Test](./dataset_scripts#ee3d-s-test) is present. Please run, 

```bash
python evaluate_ee3d_s.py 
```

The provided [pretrained](#pretrained-model) checkpoint gives us an accuracy of,

| Arch | Head_MPJPE | Neck_MPJPE | Right_shoulder_MPJPE | Right_elbow_MPJPE | Right_wrist_MPJPE | Left_shoulder_MPJPE | Left_elbow_MPJPE | Left_wrist_MPJPE | Right_hip_MPJPE | Right_knee_MPJPE | Right_ankle_MPJPE | Right_foot_MPJPE | Left_hip_MPJPE | Left_knee_MPJPE | Left_ankle_MPJPE | Left_foot_MPJPE | MPJPE | Head_PAMPJPE | Neck_PAMPJPE | Right_shoulder_PAMPJPE | Right_elbow_PAMPJPE | Right_wrist_PAMPJPE | Left_shoulder_PAMPJPE | Left_elbow_PAMPJPE | Left_wrist_PAMPJPE | Right_hip_PAMPJPE | Right_knee_PAMPJPE | Right_ankle_PAMPJPE | Right_foot_PAMPJPE | Left_hip_PAMPJPE | Left_knee_PAMPJPE | Left_ankle_PAMPJPE | Left_foot_PAMPJPE | PAMPJPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EgoHPE | 29.020 | 44.788 | 51.028 | 98.415 | 144.434 | 53.148 | 103.996 | 141.923 | 91.309 | 146.183 | 210.144 | 224.728 | 87.292 | 141.563 | 210.710 | 219.027 | 124.857 | 50.010 | 47.727 | 50.475 | 86.859 | 131.602 | 53.526 | 90.658 | 127.851 | 74.953 | 98.317 | 116.803 | 129.032 | 75.543 | 96.434 | 119.562 | 132.058 | 92.588 |


#### EE3D-R
For evaluation, ensure [EE3D-R](./dataset_scripts#ee3d-r) is present. Please run, 

```bash
python evaluate_ee3d_r.py 
```

The provided [pretrained](#pretrained-model) checkpoint gives us an accuracy of,

| Arch | walk_MPJPE | crouch_MPJPE | pushup_MPJPE | boxing_MPJPE | kick_MPJPE | dance_MPJPE | inter. with env_MPJPE | crawl_MPJPE | sports_MPJPE | jump_MPJPE | MPJPE | walk_PAMPJPE | crouch_PAMPJPE | pushup_PAMPJPE | boxing_PAMPJPE | kick_PAMPJPE | dance_PAMPJPE | inter. with env_PAMPJPE | crawl_PAMPJPE | sports_PAMPJPE | jump_PAMPJPE | PAMPJPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EgoHPE | 70.881 | 163.840 | 97.886 | 136.571 | 103.724 | 88.877 | 103.191 | 109.714 | 101.020 | 97.320 | 107.302 | 52.113 | 99.483 | 75.530 | 104.667 | 86.055 | 71.968 | 70.859 | 77.949 | 77.827 | 80.179 | 79.663 |

## Citation

If you find this code useful for your research, please cite our paper:
```
@inproceedings{Millerdurai_EventEgo3D_2024, 
    title={EventEgo3D: 3D Human Motion Capture from Egocentric Event Streams}, 
    author={Christen Millerdurai and Hiroyasu Akada and Jian Wang and Diogo Luvizon and Christian Theobalt and Vladislav Golyanik}, 
    booktitle = {Computer Vision and Pattern Recognition (CVPR)}, 
    year={2024} 
} 
```

## License

EventEgo3D is under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. The license also applies to the pre-trained models.

## Acknowledgements

The code is partially adapted from [here](https://github.com/microsoft/human-pose-estimation.pytorch). 

