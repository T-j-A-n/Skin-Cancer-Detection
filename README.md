# Skin Cancer Classification with Explainability

This project is a **Streamlit-based web application** for skin cancer image classification using a deep learning model, with **explainability through Grad-CAM and LIME**.

The app allows users to upload a skin lesion image, view the predicted disease class, and understand **why** the model made that prediction using visual explanations.

---

## Features

-  Upload skin lesion images (JPG / PNG)
-  Deep learning model (Keras, 9-class classifier)
-  **Grad-CAM** visualization  
- Red regions indicate highest contribution to prediction
-  **LIME** visualization  
- Superpixel-based local explanation
-  Side-by-side comparison of Grad-CAM and LIME
-  Displays predicted disease and confidence
-  “Disease” label shown clearly at the bottom

---

##  Disease Classes

The model predicts one of the following 9 classes:

1. Melanoma  
2. Nevus  
3. Basal Cell Carcinoma  
4. Actinic Keratosis  
5. Benign Keratosis  
6. Dermatofibroma  
7. Vascular Lesion  
8. Squamous Cell Carcinoma  
9. Unknown  

---

## Tech Stack

- **Frontend**: Streamlit  
- **Model**: TensorFlow / Keras  
- **Explainability**:
  - Grad-CAM (Gradient-weighted Class Activation Mapping)
  - LIME (Local Interpretable Model-Agnostic Explanations)
- **Libraries**:
  - NumPy
  - Pillow
  - Matplotlib
  - scikit-image

---

##  Installation

### Clone the repository
```bash
git clone https://github.com/your-username/skin-cancer-explainability.git
cd skin-cancer-explainability
```

### Install Dependencies
```bash
pip install streamlit tensorflow numpy pillow matplotlib lime scikit-image
```

### Running the Application
Make sure the model file is present:
````bash
my_model.keras
````

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open the browser at:
```bash
http://localhost:8501
```

