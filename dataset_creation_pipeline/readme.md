# Synthetic Data Pipeline for SHREC 2021

This project implements a data pipeline for generating synthetic multimodal datasets based on the **SHREC 2021: Retrieval of Cultural Heritage Objects** challenge.

The dataset focuses on **3D representations of Peruvian vessels**, combining geometric data, rendered images, and text descriptions.

---

## Dataset Structure

The generated dataset follows a hierarchical multimodal structure:

```
dataset/
│
├── object_1/
│   ├── image_1/
│   │   ├── text_1.txt
│   │   ├── text_2.txt
│   │   └── ...
│   │
│   ├── image_2/
│   │   └── ...
│
├── object_2/
│   └── ...
```


Each object contains:
- Multiple **rendered images**
- Multiple **text descriptions per image**

---

## Dataset Composition

The dataset is built from two main subsets:

- **Shape Set**: Contains all 3D objects based on geometry.
- **Culture Set**: A subset of the shape set with additional cultural metadata.

A key step in the pipeline identifies overlapping elements and merges them into a unified representation.

---

## Data Splits

The pipeline automatically generates three splits:

- **Train**
- **Validation**
- **Test**

Although the original datasets include Train/Test splits, they are:
1. Merged  
2. Processed jointly  
3. Re-split to include a validation set  

---

## Pipeline Overview

The dataset is generated through the following steps:

1. **IndexerMerger**  
   Merges indices from different dataset sources.

2. **LabelJSONCreation**  
   Converts labels into a structured JSON format.

3. **LabelMerger**  
   Unifies labels across datasets.

4. **CloudCreation**  
   Generates 3D point clouds from mesh data.

5. **Render**  
   Produces 2D images from 3D objects.

6. **DatasetCreatorGemini**  
   Generates textual descriptions for each image.

7. **DatasetMerger**  
   Combines all modalities into the final dataset structure.

---

## Intermediate Outputs

Each pipeline step produces its own output folder for easier debugging and validation:

- `cloud/` → Point clouds  
- `images/` → Rendered images  
- `texts/` → Generated descriptions  

These are later merged into the final dataset.

---

## Purpose

This pipeline is designed to support multimodal learning tasks, especially:

- 3D–Text retrieval  
- Cross-modal representation learning  
- Cultural heritage understanding  

## 📄 License

This project includes the `render.py` script, which uses a modified version of the `render_blender.py` script from the **stanford-shapenet-renderer**.

The original renderer is distributed under its own license, which is included in this repository as `LICENSE.txt`.

This project does not claim ownership of that component. All rights and credits belong to the original authors.
