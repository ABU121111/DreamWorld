# DreamWorld: Unified World Modeling in Video Generation

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2603.00466)
> **DreamWorld** is a unified framework that integrates complementary world knowledge into video generators via a Joint World Modeling Paradigm.<br>
Boming Tan, Xiangdong Zhang, Ning Liao, Yuqing Zhang, Shaofeng Zhang, Xue Yang, Qi Fan, Yanyong Zhang

## 📰 News

- 🎉 Sept, 2025: Our work [**VideoREPA**](https://github.com/aHapBean/VideoREPA) is accepted by NeurIPS 2025, the first adaptation of REPA to video generation, transferring physical knowledge via Token Relation Distillation to improve the physical realism of text-to-video models.
- 💥 Feb, 2026: [**DreamWorld**](https://github.com/ABU121111/DreamWorld) is available on [Arxiv](https://arxiv.org/abs/2603.00466).

## 📖 Overview

Despite impressive progress in video generation, existing models remain limited to surface-level plausibility and lack a coherent, unified understanding of the world. Prior approaches typically incorporate only a single form of world-related knowledge or rely on rigid alignment strategies. 

To address this limitation, we introduce **DreamWorld**, which jointly predicts video pixels and features from foundation models to capture temporal dynamics, spatial geometry, and semantic consistency.

## ✨ Key Features

* **Joint World Modeling Paradigm**: Integrates temporal dynamics from Optical Flow, spatial geometry from VGGT, and semantic understanding from DINOv2.
* **Consistent Constraint Annealing (CCA)**: A progressive decay mechanism that regulates world-level constraints during training to mitigate visual instability and temporal flickering, ensuring high-fidelity generation.
* **Multi-Source Inner-Guidance**: Leverages the model's own predicted knowledge features during inference to steer the generation process, ensuring trajectories strictly adhere to real-world laws.

## 🏗️ Architecture

Below is the overview of the DreamWorld training and inference pipeline:

<img width="6459" height="2424" alt="Train" src="https://github.com/user-attachments/assets/b472e2aa-cbaa-4e24-9a39-239f15a52200" />

## 📊 Quantitative Results

Extensive evaluations show that DreamWorld significantly outperforms baselines and establishes a new standard for world models.

### VBench Evaluation
DreamWorld demonstrates significant improvements over baselines, particularly in temporal dynamics, semantic understanding, and spatial relationships. **FT.** denotes the fine-tuned version, and **Reimpl.** indicates our re-implementation of the method.

| Method | Quality Score | Semantic Score | Overall Score |
| :--- | :--- | :--- | :--- |
| Wan2.1-T2V-1.3B | 79.81 | 65.43 | 76.93 |
| Wan2.1-T2V-1.3B(FT.) | 81.26 | 68.47 | 78.71 |
| VideoJAM(Reimpl.) | 81.18 | 69.08 | 78.76 |
| **DreamWorld (Ours)** | **83.49** | **70.89** | **80.97** |


## 🎥 Comparison Videos

<table>
  <tr>
    <th width="27%" align="center">Wan2.1(FT.)</th>
    <th width="27%" align="center">VideoJAM(Reimpl.)</th>
    <th width="30%" align="center">DreamWorld (Ours)</th>
    <th width="16%" align="center">Prompt</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/36a02d9b-2dd6-40b5-ae7e-1c6e47cac746" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/0ba95548-a249-4a79-b057-42a4034401c6" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/6b35a1ce-375e-4077-a852-074a216b35b3" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">Gwen Stacy reading a book, tilt up.</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/8cc6d651-4c25-49c6-a46a-723ffd5a9dd4" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/2dfa0e51-538c-405f-991d-94f7b68a4e53" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/2d6b9e02-f866-4e1c-bc20-dc91d97f2f59" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">A hose sprays water onto a burning pile of tires...</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/d3f16d1a-91ca-40b3-9be8-ad0321a763da" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/96b5c5f6-3147-4566-a46a-88ec06504eac" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/cb5fadf2-423c-4b9b-8df5-84a5583b1768" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">A person wades through a swamp...</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/71f57bc0-11c2-4fba-bf6a-7844d09da597" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/16609298-48d6-4275-8e48-2191d349b8cb" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/3f5311f4-e1dd-4ff9-b4e9-cd0bc17268c9" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">Happy dog wearing a yellow turtleneck...</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/7fd1ccba-174e-4cb5-8679-6a56894d447a" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/6a619dbf-1c8b-4244-bd34-070da42d094b" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/35124171-d49a-4a2f-8467-bc71ecbddea3" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">  A potter's wheel is lightly poked with a tool...</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/3663d950-13e9-4057-a5fa-d48a5ebe5da5" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/04c1ff3f-ae70-4644-89ce-99bb01fe8340" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/bb201cba-4dbe-419f-a97d-789bd3338941" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">Two roller skaters perform a synchronized spin...</td>
  </tr>
</table>
