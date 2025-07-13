# **PSFHS-DataAugmentation**

A project for training a UNet-based segmentation model on ultrasound images with PyTorch, including data augmentation and cross-validation.

---

## **Table of Contents**

[Project Description](#project-description)

[Requirements](#requirements)

[Installation](#installation)

[Data Structure](#data-structure)

[Quick Start](#quick-start)

[Code Structure](#code-structure)

[Reproducibility](#reproducibility)

[Acknowledgements](#acknowledgements)

[Contact](#contact)

---

## **Project Description** {#project-description}

This project provides code for medical image segmentation using a UNet architecture, leveraging **PyTorch** and **Albumentations** for data augmentation. Includes tools for cross-validation, evaluation (Dice, IoU), and visualization.

---

## **Requirements** {#requirements}

* Python \>= 3.8

* PyTorch \>= 1.10

* albumentations

* numpy

* matplotlib

* SimpleITK

* scikit-learn

Install via pip:

`pip install torch albumentations numpy matplotlib SimpleITK` 

`scikit-learn`

---

## **Installation** {#installation}

**Clone this repository:**

	`git clone https://github.com/aijaz808/PSFHS-DataAugmentation.git`

`cd PSFHS-DataAugmentation`

**Install requirements:**

	`pip install -r requirements.txt` 

**`Dataset dependency:`**

	`We would need to download the following dataset after taking the pull of main branch.`

`Link to dataset : https://zenodo.org/records/10969427`

	

---

## **Data Structure** {#data-structure}

Expected directory layout:

`PSFHS-DataAugmentation/`  
    `├── image_mha/`  
    `│     ├── 00001.mha`  
    `│     └── ...`  
    `├── label_mha/`  
    `│     ├── 00001.mha`  
    `│     └── ...`  
    `└── main.py`

* **image\_mha/**: Input images (.mha format)

* **label\_mha/**: Corresponding ground-truth masks

---

## **Quick Start** {#quick-start}

**Edit** your image and mask directory if needed (`image_mha/*.mha`, `label_mha/*.mha`).

**Run the training script** (adapt as needed):

`python main.py`  
---

## **Code Structure** {#code-structure}

* `main.py` — Main training script.

* **Key components:**

  * `set_seed()` for reproducibility

  * Dataset definition using SimpleITK for .mha reading

  * Data loading and augmentation (using albumentations)

  * UNet model definition

  * Training and validation loops

  * Metrics (Dice, IoU)

  * Visualization of results

---

## **Reproducibility** {#reproducibility}

The script sets seeds for torch, numpy, and random for reproducibility.

---

## **Acknowledgements** {#acknowledgements}

* [Albumentations](https://albumentations.ai/)

* [PyTorch](https://pytorch.org/)

* [SimpleITK](https://simpleitk.readthedocs.io/)

* [scikit-learn](https://scikit-learn.org/)

---

## **Contact** {#contact}

For questions or feedback, please open an issue or contact \[your email here\].

