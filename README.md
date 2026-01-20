# CSI-HAR — Human Activity Recognition using WiFi CSI

## 📌 Project Overview

This project aims to develop an **intelligent Human Activity Recognition (HAR) system**
based on **WiFi Channel State Information (CSI)** by combining:

- a **software development part (website / backend)**  
- a **Deep Learning part based on Convolutional Neural Networks (CNNs)**  

The system automatically recognizes human activities **without using cameras or wearable sensors**, ensuring user privacy.

---

## 🎯 Project Objectives

- Exploit WiFi CSI signal variations caused by human motion
- Transform raw CSI data into meaningful representations
- Design and train a Deep Learning model for activity classification
- Integrate the trained model into an application backend
- Provide an end-to-end automated activity recognition pipeline

---

## 🏗️ Global Architecture

The project is organized into **two complementary components**:

1. **Development Component (Website / Backend)**
2. **Deep Learning Component (CNN Model)**

These components interact to form a complete pipeline, from CSI data input to final activity prediction.

---

## 🌐 Development Component (Website / Backend)

The development part consists of building a **backend application** that:

- handles CSI data uploads (CSV files),
- preprocesses and prepares the data,
- converts CSI signals into model-compatible formats,
- communicates with the Deep Learning model,
- returns and displays prediction results.

This component acts as the **interface between the user and the AI model**, orchestrating all processing steps.

---

## 🧠 Deep Learning Component (Human Activity Recognition)

### 🔹 Principle

Raw CSI data is noisy and difficult to interpret directly.
To overcome this, the project adopts a **Deep Learning approach using CNNs**.

The process includes:
1. CSI data preprocessing
2. Transformation of CSI signals into image-like representations
3. Automatic feature extraction using CNNs
4. Activity classification

---

## 🤖 CNN Model Developed

The model developed in this project is based on:

- **MobileNetV2**
- a **lightweight and efficient CNN architecture**
- **Transfer Learning**

### Why MobileNetV2?
- High performance with low computational cost
- Fast inference
- Suitable for real-world and application-level deployment

---

## 🎯 Role of the Developed Model

The CNN model is used to:

- analyze WiFi signal variations caused by human movements,
- automatically extract discriminative features,
- classify human activities,
- provide predictions along with confidence scores.

---

## 🧍 Recognized Activities

The system is capable of recognizing the following activities:

- Walk
- Run
- Fall
- Sit down
- Stand up
- Bend
- Lie down

---

## 📂 Project Structure

CSI-HAR/
├── backend/
│ ├── app.py
│ ├── build_dataset_from_csv.py
│ ├── generate_graph.py
│ ├── generate_test_csvs.py
│ ├── make_statistics.py
│ ├── predict_csi_har.py
│ ├── test_upload.py
│ ├── train_from_folders.py
│ ├── classes.json
│ └── csi_results.json
│
├── README.md
├── requirements.txt
└── .gitignore


---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Generate sample CSI data
```bash
python backend/generate_test_csvs.py
```
### 3️⃣ Train the model
```bash
python backend/train_from_folders.py
```
### 4️⃣ Predict an activity
```bash
python backend/predict_csi_har.py
```
📊 Evaluation and Results

The model is evaluated using:

accuracy

loss curves

confusion matrix

classification report

Experimental results demonstrate that WiFi CSI data can effectively capture human motion patterns and enable reliable recognition of daily activities.

Trained models and datasets are not included in the repository due to size constraints.

🔐 Privacy Considerations

This system does not use cameras, microphones, or wearable sensors.
All activity recognition is performed using WiFi signal variations, ensuring privacy preservation.

🎓 Academic Context

This project was developed in an academic context to explore:

Deep Learning techniques

Wireless sensing

Human Activity Recognition

Integration of AI models into software systems
