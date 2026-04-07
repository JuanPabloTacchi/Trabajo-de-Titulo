# Multimodal Models Comparison (Image + Text + 3D)  
## Content Retrieval for Archaeological Vessels

## Description

This work focuses on the development and evaluation of a **multimodal model (image + text + 3D)** for **content retrieval of archaeological vessels**.

The main goal is to learn joint representations that enable retrieving relevant information from different types of queries, including text, images, and 3D geometry.

---

## Objectives

1. Develop a **multimodal dataset** containing:
   - Images  
   - Text descriptions  
   - 3D representations (point clouds)

2. Perform **fine-tuning** on multimodal models for content retrieval tasks.

3. Evaluate the effectiveness of the models in terms of:
   - Ability to retrieve relevant information

4. Evaluate **inference time** and **memory usage**:
   - Measure the time and memory required to perform queries

---

## Project Structure

This project is composed of two main parts:

### 1. Dataset Creation

- Based on Peruvian vessels from:  
  **SHREC 2021: Retrieval of Cultural Heritage Objects**
- Includes processing and alignment of:
  - 3D data  
  - Images  
  - Text

To reproduce this process follow the readme in:

    dataset_creation_pipeline/

> ⚠️ The dataset has already been created but is not yet publicly available.

---

### 2. Multimodal Model Fine-Tuning

- A fork of the **Uni3D (BAAI-Vision)** model is used
- Different variants of the model are fine-tuned for multimodal retrieval tasks

> Note: The original model license is included in the repository.

> Learn more of Uni3D at https://github.com/baaivision/Uni3D

> Trained models are not yet publicly available.

---

## Installation

To set up the environment on Linux:

```bash
bash create_uni3d.sh
```

This script installs all required dependencies to run the fine-tuned models.

---

## Scripts

This repository provides several scripts for training, evaluation, and analysis of the models:

- **`embedd_dataset.sh`**:  
  Generates an embedded dataset from the Vessels Dataset generated in the pipeline using a specified `pretrain_dataset_name` (Greatly optimizes the Finetuning).  

- **`individual_test.sh`**:  
  Runs an evaluation for a single query against the embedding space.  
  > The input text must be manually modified inside the `individual_test()` function in `main.py`.

- **`memory_test.sh`**:  
  Measures GPU memory usage and inference time during model execution.

- **`pretrain_memoria_t_to_g.sh`**:  
  Fine-tunes Uni3D model variants, ranging from **tiny to giga** configurations.

- **`save_model.sh`**:  
  Converts a distributed-trained model into a single `.pt` file for easier loading.

- **`test_precision.sh`**:  
  Runs the full evaluation pipeline, including:
  - Top-1, Top-3, Top-5, Top-10 accuracy  
  - Mean Reciprocal Rank (MRR)

