# Car Damage Severity Classification  

**GROUP ID: 36**  
**Ontario Tech University – Machine Learning Course Project (Winter 2025)**
**Group Members:**  

- [Arian Vares] 	      – arian.vares@ontariotechu.net         -  Student ID: 100882708
- [Yash Patel]          – yash.patel14@ontariotechu.net      -  Student ID: 100785833 
- [Sayen Mayuran]  – sayen.mayuran@ontariotechu.net  -  Student ID: 100xxxxxx  


## Project Overview  
This project implements an image classification system that automatically determines the severity of vehicle damage from a single photograph. The model outputs one of three classes:  
- **01-minor**  
- **02-moderate**  
- **03-severe**  

The solution is designed for real-world use in insurance claim automation, repair shop triage, and rapid accident assessment.

## Dataset  
Source: [Car Damage Severity Dataset (Kaggle)](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)  
- Training set: 1,383 images  
- Validation set: 248 images  
- Images are pre-organized into `data3a/training` and `data3a/validation` folders by severity class.  
All experiments use a fixed input size of 256×256 pixels.

## Repository Structure

├── 01_data_loading_and_eda.ipynb                     ← Data loading, exploration and visualisation

├── 02_transfer_learning_vgg16_frozen_backbone.ipynb  ← Frozen VGG16 baseline (88.1 % val acc)

├── 03_finetune_vgg16.ipynb                            ← Complete fine-tuning pipeline + callbacks + evaluation

├── app.py                                             ← Production-ready Streamlit deployment

├── requirements.txt                                   ← Environment dependencies

├── models/                                            ← Saved models (best checkpoint + final)

└── README.md


## Implemented Models & Results  

| Model                          | Backbone State | Validation Accuracy | Key Observations                              |
|--------------------------------|----------------|---------------------|-----------------------------------------------|
| VGG16 (frozen backbone)        | Frozen         | 88.1 %              | Stable baseline, fast convergence             |
| VGG16 (fine-tuned, last 4 blocks) | Partially trainable | **94.8 %**       | Best overall performance on validation set   |
| Custom CNN (from scratch)      | –              | ~94–95 %            | Excellent training curves, very stable        |

The fine-tuned VGG16 model was selected for deployment due to its strong generalisation and clean training history.

## Training Details (03_finetune_vgg16.ipynb)  
- Two-phase training:  
  1. Frozen backbone – head only (20 epochs)  
  2. Fine-tuning – last 4 convolutional blocks unfrozen, learning rate 1e-5  
- Callbacks used: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  
- Final evaluation includes classification report and confusion matrix (saved as PNG)  
- Model exported in both Keras (`.keras`) and SavedModel directory formats  

## Deployment  
A fully functional web application is provided in `app.py` (Streamlit). Users can:  
- Upload a JPEG/PNG image of a damaged vehicle  
- Instantly receive the predicted severity class with confidence score  
- View a probability bar chart for all three classes  

The app.py automatically detects and loads either a `.keras/.h5` file or a SavedModel directory, making it robust for future model updates.

## How to Run the Demo Locally  
```bash
git clone https://github.com/SayenM/CSCI-4050U---Car-Damage-Severity.git
cd CSCI-4050U---Car-Damage-Severity
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py