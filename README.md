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
    <td><video src="https://github.com/user-attachments/assets/36a02d9b-2dd6-40b5-ae7e-1c6e47cac746" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/0ba95548-a249-4a79-b057-42a4034401c6" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/6b35a1ce-375e-4077-a852-074a216b35b3" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">Gwen Stacy reading a book, tilt up.</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/487ee0b8-7ba6-4281-9550-07e7ec4daddd" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/e123ef42-0d6a-41a2-a8a0-e67ed8402ab2" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/5589d7d7-18db-42a3-9c1d-8c0d0a64b82d" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">A person is picking up the phone.</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/f6f48d08-f629-46e1-8b81-6c03c91d65b0" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/e2ae9f2d-50ed-4bb8-b2c1-0cda19df8a9a" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/eea8924e-171f-4eb0-b865-97b16efc743a" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">A cup of tea is tilted in the space station...</td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/71f57bc0-11c2-4fba-bf6a-7844d09da597" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/16609298-48d6-4275-8e48-2191d349b8cb" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/3f5311f4-e1dd-4ff9-b4e9-cd0bc17268c9" autoplay loop muted playsinline width="100%"></video></td>
    <td align="center" valign="middle">Happy dog wearing a yellow turtleneck...</td>
  </tr>
</table>
