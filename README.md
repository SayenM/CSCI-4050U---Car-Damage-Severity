# Car Damage Severity Classification  
**GROUP ID: 36**
**Ontario Tech University – Machine Learning Course Project (Winter 2025)**  
**Group Members:**  

- [Arian Vares] 	      – arian.vares@ontariotechu.net         -  Student ID: 100882708
- [Sayen Mayuran]  – sayen.mayuran@ontariotechu.net  -  Student ID: 100xxxxxx  
- [Yash Patel]          – yash.patel14@ontariotechu.net      -  Student ID: 100xxxxxx  


## Project Overview
We are building a deep learning model that can automatically classify the severity of car damage from images into three classes:  
- 01-minor  
- 02-moderate  
- 03-severe  

This has real-world applications in insurance claim automation and repair cost estimation.

## Dataset
- Source: Car Damage Severity Dataset (Kaggle)  
  https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset  
- Training folder: 1,383 images  
- Validation folder: 248 images  
- Images are already organized into subfolders by class (data3a/training and data3a/validation)

## Current Repository Contents
├── 01_data_loading_and_eda.ipynb    ← Dataset loading, verification and sample visualization    

├── README.md                         ← This file


The notebook 01_data_loading_and_eda.ipynb contains:  
- Successful loading of the dataset using tf.keras.utils.image_dataset_from_directory  
- Verification of class names and folder structure  
- Visualization of 16 random training images with correct labels  
- Initial train/validation/test split preparation  

## Next Steps (in progress)
1. Build an optimized tf.data pipeline with augmentation and caching  
2. Implement transfer learning (starting with VGG16, later ResNet50/EfficientNet)  
3. Two-phase training: frozen backbone → fine-tuning  
4. Export the final model and create a Streamlit web demo  
5. Record the required 10-minute YouTube demonstration video  

We will add the remaining notebooks and the live demo over the next few days.


