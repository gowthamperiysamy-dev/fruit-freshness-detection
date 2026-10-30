# Fruit Freshness Detection Using MobileNetV2

## Project Overview

The Fruit Freshness Detection System is a deep learning-based application that identifies whether a fruit is Fresh or Rotten by analyzing its image.

The system uses MobileNetV2, a pretrained convolutional neural network, through transfer learning. A Streamlit web interface allows users to upload fruit images and receive predictions with confidence scores.

## Technologies Used

- Python 3.11
- TensorFlow 2.21
- MobileNetV2
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow

## Dataset

The project uses a Fresh and Rotten Fruits Image Dataset containing:

- 8 fruit categories
- 1,600 fresh fruit images
- 1,600 rotten fruit images
- 3,200 total images

The dataset is converted into two classification categories:

- Fresh
- Rotten

Dataset split:

| Dataset | Percentage | Images |
|---|---:|---:|
| Training | 70% | 2,240 |
| Validation | 15% | 480 |
| Testing | 15% | 480 |

Dataset source: [Mendeley Data](https://data.mendeley.com/datasets/bdd69gyhv8/1)

## Model Architecture

MobileNetV2 is used as the feature extraction backbone.

The classification architecture consists of:

1. Input image resizing (160 × 160).
2. Image normalization.
3. Pretrained MobileNetV2 feature extraction.
4. Global Average Pooling.
5. Dropout layer.
6. Sigmoid output layer for binary classification.

The pretrained MobileNetV2 base layers are frozen during initial training.

## Model Performance

The model achieved the following results on the held-out test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 93.96% |
| Precision (Rotten) | 94.89% |
| Recall (Rotten) | 92.92% |
| F1-Score (Rotten) | 93.89% |

## Confusion Matrix

The confusion matrix evaluates how accurately the model distinguishes fresh and rotten fruits.

## Features

- Upload fruit images.
- Predict Fresh or Rotten classification.
- Display prediction confidence.
- Display class probabilities.
- Evaluate model using accuracy, precision, recall and F1-score.
- Visualize the confusion matrix.
- Display training accuracy graph.

## Project Structure

```text
fruit-freshness-detection/
│
├── fruit_dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── prepare_dataset.py
├── train_model.py
├── app.py
├── run_app.bat
├── requirements.txt
├── .gitignore
├── README.md
│
├── fruit_freshness_model.keras
├── best_model.keras
├── metrics.json
├── confusion_matrix.png
└── accuracy_graph.png
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fruit-freshness-detection.git
cd fruit-freshness-detection
```

Create a virtual environment:

```bash
py -3.11 -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Training the Model

To train the model again:

```bash
python train_model.py
```

Ensure the dataset is arranged in the required directory structure before training.

## Future Enhancements

- Support for additional fruit categories.
- Improve classification using fine-tuning.
- Deploy the application online.
- Test under different lighting and background conditions.
- Extend the model to estimate fruit quality and spoilage severity.

## Disclaimer

This application predicts visual freshness based on image patterns. It does not determine food safety or guarantee that a fruit is safe to consume.
