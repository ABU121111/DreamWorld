# DreamWorld: Unified World Modeling in Video Generation


> **DreamWorld** is a unified framework that integrates complementary world knowledge into video generators via a Joint World Modeling Paradigm. 

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
DreamWorld demonstrates significant improvements over baselines, particularly in temporal dynamics, semantic understanding, and spatial relationships. **FT** denotes the fine-tuned version, and **Reimpl.** indicates our re-implementation of the method.

| Method | Quality Score | Semantic Score | Overall Score |
| :--- | :--- | :--- | :--- |
| Wan2.1-T2V-1.3B | 79.81 | 65.43 | 76.93 |
| Wan2.1-T2V-1.3B(FT) | 81.26 | 68.47 | 78.71 |
| VideoJAM(Reimpl.) | 81.18 | 69.08 | 78.76 |
| **DreamWorld (Ours)** | **83.49** | **70.89** | **80.97** |

## 🎥 Comparison Videos


<table>
  <tr>
    <th width="27%" align="center">Wan2.1</th>
    <th width="27%" align="center">VideoJAM</th>
    <th width="27%" align="center">DreamWorld (Ours)</th>
    <th width="19%" align="center">Prompt</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/735bfabb-7bef-484b-ace1-9d86ece2c00f" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/65a4e666-6a3d-4a1d-9290-8478d8e6cf04" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/cdae8cca-1bbc-4234-9594-a9bdf26fc48e" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">Gwen Stacy reading a book, tilt up.</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/7496e3cf-a150-4d86-a450-67f9ebe935d1" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/10678aca-a32b-439f-83ef-a32686f4beae" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/68936096-2d75-41dc-ac36-5a29bd7e1eb4" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">A person is picking up the phone.</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/ac194c14-7678-4f53-8656-11479dd119f7" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/4ea9ff9e-b0b6-42cd-ac01-4acbe0fa1af0" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/432ac3d4-7fb1-4611-8e45-dbf2cb968989" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">A cup of tea is carefully tilted in the space station, and the liquid floats in various directions.</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/6465a869-2297-4b42-ba65-8cb0fa074e6e" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/b935e1a5-967e-4dc7-aaff-a55117fe7c23" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/36a01fd3-0a54-426f-8baf-43536b67bba9" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">Happy dog wearing a yellow turtleneck, studio, portrait, facing camera, dark background</td>
  </tr>
</table>
