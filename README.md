# Car Damage Severity Classification  

**GROUP ID: 36**  
**Ontario Tech University – Machine Learning Course Project (Winter 2025)**
**Group Members:**  

- [Arian Vares] 	      – arian.vares@ontariotechu.net         -  Student ID: 100882708
- [Yash Patel]          – yash.patel14@ontariotechu.net      -  Student ID: 100785833 
- [Sayen Mayuran]  – sayen.mayuran@ontariotechu.net  -  Student ID: 100xxxxxx  


## Project Description  
We are developing a deep learning model that automatically predicts the severity of car damage from a single image. The model classifies damage into three categories:  
- 01-minor  
- 02-moderate  
- 03-severe  

This has practical applications in insurance claim processing and repair-cost estimation.

## Dataset  
- Source: [Car Damage Severity Dataset – Kaggle](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)  
- Training: 1,383 images  
- Validation: 248 images  
- Already organized into `data3a/training` and `data3a/validation` folders by class

## Project Description  
We are developing a deep learning model that automatically predicts the severity of car damage from a single image. The model classifies damage into three categories:  
- 01-minor  
- 02-moderate  
- 03-severe  

This has practical applications in insurance claim processing and repair-cost estimation.

## Dataset  
- Source: [Car Damage Severity Dataset – Kaggle](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)  
- Training: 1,383 images  
- Validation: 248 images  
- Already organized into `data3a/training` and `data3a/validation` folders by class

## Repository Contents 
├── 01_data_loading_and_eda.ipynb                          ← Dataset loading, verification, sample visualization

├── 02_transfer_learning_vgg16_frozen_backbone.ipynb       ← Phase 1: VGG16 (frozen backbone) + data augmentation
→ Achieved 88.1% validation accuracy after 20 epochs

├── README.md

## Current Progress  
- Dataset loading and exploration completed  
- Transfer learning implemented using pre-trained VGG16 (backbone frozen)  
- Data augmentation and normalization pipeline added  
- Full training of classification head completed (88.1% validation accuracy)

## To Be Completed
- Fine-tuning of VGG16 (unfreeze last few blocks + low learning rate)  
- Add callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)  
- Final evaluation + confusion matrix  
- Export model as SavedModel  
- Build Streamlit demo app (upload image → prediction)  
- Create presentation slides (PDF export)  
- Record 8–10 minute YouTube demonstration video  

All code needed for fine-tuning and the rest is already in notebook 02 — just change `trainable = True` for the last blocks and re-run.

Last updated: December 5, 2025  
Project on track for completion by December 6.
