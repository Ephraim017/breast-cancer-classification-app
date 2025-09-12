# Breast Cancer Classification App# Breast Cancer Classification Web App



A comprehensive web application for breast cancer histological image classification using deep learning. This app uses a fine-tuned ResNet50 model trained on the BreakHis dataset to classify histological images as benign or malignant.A Flask-based web application that uses AI (Convolutional Neural Network) to classify breast histological images as benign or malignant.



## 🎯 Overview## Features



This application provides an intuitive web interface for medical professionals and researchers to classify breast cancer histological images. The model achieves **98.2% validation accuracy** and provides confidence scores and risk assessments for each prediction.- **AI-Powered Classification**: Uses a CNN model trained on the BreakHis dataset

- **User-Friendly Interface**: Modern, responsive web design

## 🧠 Model Performance- **Real-time Analysis**: Upload and get results instantly

- **Detailed Results**: Confidence scores, probability breakdown, and risk assessment

- **Architecture**: ResNet50 with transfer learning from ImageNet- **Medical Disclaimer**: Clear warnings about research-only usage

- **Dataset**: BreakHis (7,909 histological images)- **API Support**: RESTful API for programmatic access

- **Training Accuracy**: 98.2%

- **Validation Accuracy**: 98.2% ## Installation

- **Parameters**: 24.1M trainable parameters

- **Training Time**: 45 epochs with early stopping### Local Development



## 🚀 Features1. **Clone the repository**

```bash

- **Real-time Image Classification**: Upload histological images for instant classificationgit clone <repository-url>

- **Confidence Scoring**: Provides probability scores for both benign and malignant classificationscd breast-cancer-app

- **Risk Assessment**: 5-level risk categorization (Very Low to Very High Risk)```

- **Medical-grade Interface**: Clean, professional UI suitable for clinical environments

- **Batch Processing**: Support for multiple image uploads2. **Create virtual environment**

- **Detailed Logging**: Comprehensive prediction logging for audit trails```bash

python -m venv venv

## 🛠️ Technical Stacksource venv/bin/activate  # On Windows: venv\Scripts\activate

```

- **Backend**: Flask (Python)

- **Machine Learning**: TensorFlow/Keras3. **Install dependencies**

- **Frontend**: HTML5, CSS3, JavaScript```bash

- **Model**: ResNet50 with custom classification headpip install -r requirements.txt

- **Image Processing**: PIL (Pillow), NumPy```



## 📋 Prerequisites4. **Run the application**

```bash

- Python 3.8+python app.py

- TensorFlow 2.x```

- Flask

- PIL (Pillow)5. **Access the application**

- NumPyOpen your browser and go to `http://localhost:5000`



## 🔧 Installation### Docker Deployment



1. **Clone the repository**:1. **Build and run with Docker Compose**

   ```bash```bash

   git clone https://github.com/yourusername/breast-cancer-classification-app.gitdocker-compose up --build

   cd breast-cancer-classification-app```

   ```

2. **Access the application**

2. **Create virtual environment**:Open your browser and go to `http://localhost:5000`

   ```bash

   python -m venv venv### Production Deployment with Nginx

   source venv/bin/activate  # On Windows: venv\Scripts\activate

   ```1. **Run with production profile**

```bash

3. **Install dependencies**:docker-compose --profile production up --build

   ```bash```

   pip install -r requirements.txt

   ```2. **Access the application**

Open your browser and go to `http://localhost`

4. **Download the trained model**:

   - Place the `resnet50_breast_cancer_final.keras` model file in the root directory## Project Structure

   - The model file is ~290MB and should be downloaded separately due to GitHub file size limits

```

## 🏃‍♂️ Usagebreast-cancer-app/

├── app.py                 # Main Flask application

1. **Start the application**:├── model_utils.py         # AI model utilities

   ```bash├── requirements.txt       # Python dependencies

   python app.py├── Dockerfile            # Docker configuration

   ```├── docker-compose.yml   # Docker Compose configuration

├── README.md             # This file

2. **Access the web interface**:├── templates/            # HTML templates

   - Open your browser and go to `http://localhost:5000`│   ├── base.html

│   ├── index.html

3. **Upload and classify images**:│   ├── upload.html

   - Click "Choose File" to select a histological image│   ├── result.html

   - Supported formats: PNG, JPG, JPEG│   ├── about.html

   - Click "Classify Image" to get predictions│   ├── 404.html

   - View results including confidence scores and risk assessment│   └── 500.html

├── static/              # Static files

## 🏗️ Model Architecture│   ├── css/

│   └── uploads/         # Uploaded images (created automatically)

```└── models/              # AI model files (place your trained model here)

ResNet50 Base (ImageNet weights)```

    ↓

Global Average Pooling 2D## Model Information

    ↓

Dense Layer (256 units, ReLU, L2 regularization)### Architecture

    ↓- **Type**: Convolutional Neural Network (CNN)

Dropout (0.6)- **Input Size**: 128x128x3 (RGB images)

    ↓- **Layers**: 3 Convolutional layers + 2 Dense layers

Dense Layer (2 units, Softmax) - Binary Classification- **Output**: Binary classification (Benign/Malignant)

```

### Training Dataset

## 📊 Training Details- **Dataset**: BreakHis v1

- **Images**: 7,909 microscopic images

- **Optimizer**: Adam (lr=1e-5, epsilon=1e-07)- **Classes**: Benign (2,480) and Malignant (5,429)

- **Loss Function**: Categorical Crossentropy- **Magnifications**: 40X, 100X, 200X, 400X

- **Data Augmentation**: Rotation, width/height shift, shear, zoom, horizontal flip

- **Regularization**: L2 regularization, Dropout### Performance Metrics

- **Callbacks**: Early stopping, model checkpointing, learning rate reduction- **Accuracy**: ~85%

- **Precision**: ~82%

## 🎨 Risk Level Interpretation- **Recall**: ~88%

- **AUC Score**: ~0.9

| Malignant Probability | Risk Level | Clinical Interpretation |

|----------------------|------------|------------------------|## API Usage

| < 20% | Very Low Risk | Benign characteristics, routine follow-up |

| 20-40% | Low Risk | Mostly benign, consider additional imaging |### Predict Endpoint

| 40-60% | Moderate Risk | Mixed characteristics, clinical correlation needed |

| 60-80% | High Risk | Suspicious features, recommend biopsy |**URL**: `/api/predict`  

| > 80% | Very High Risk | Highly suspicious, immediate clinical attention |**Method**: `POST`  

**Content-Type**: `multipart/form-data`

## 🐳 Docker Support

**Request**:

Build and run with Docker:```bash

curl -X POST -F "file=@image.png" http://localhost:5000/api/predict

```bash```

docker build -t breast-cancer-app .

docker run -p 5000:5000 breast-cancer-app**Response**:

``````json

{

Or use Docker Compose:  "predicted_class": "Benign",

  "confidence": 0.89,

```bash  "benign_probability": 0.89,

docker-compose up  "malignant_probability": 0.11,

```  "risk_level": "Low Risk"

}

## 📁 Project Structure```



```## Usage Guidelines

breast-cancer-classification-app/

├── app.py                 # Main Flask application### Best Practices

├── model_utils.py         # Model utilities and prediction logic- Use high-quality histological images

├── train_resnet50.py      # Training script- Ensure proper image format (PNG, JPG, JPEG, GIF, BMP, TIFF)

├── requirements.txt       # Python dependencies- Images should be well-focused and clear

├── Dockerfile            # Docker configuration- Maximum file size: 16MB

├── docker-compose.yml    # Docker Compose configuration

├── templates/            # HTML templates### Limitations

│   ├── index.html        # Main upload page- **Research Use Only**: Not for clinical diagnosis

│   └── result.html       # Results display page- Performance may vary with image quality

├── static/               # Static assets- Requires validation by medical professionals

│   ├── css/             # Stylesheets- Should not replace professional medical consultation

│   ├── js/              # JavaScript files

│   └── uploads/         # Uploaded images (created at runtime)## Security Considerations

└── sample_images/        # Sample test images

```- Uploaded images are processed securely

- Files are automatically cleaned up after processing

## 🧪 Testing- No permanent storage of uploaded medical images

- Input validation for file types and sizes

Test the application with sample images:

## Environment Variables

```bash

python -c "- `FLASK_ENV`: Set to `production` for production deployment

from model_utils import BreastCancerClassifier- `FLASK_APP`: Application entry point (default: `app.py`)

classifier = BreastCancerClassifier()- `PYTHONPATH`: Python path configuration

result = classifier.predict('sample_images/test_image.png')

print(result)## Deployment Options

"

```### 1. Local Development

```bash

## 🤝 Contributingpython app.py

```

1. Fork the repository

2. Create a feature branch (`git checkout -b feature/new-feature`)### 2. Docker (Development)

3. Commit your changes (`git commit -am 'Add new feature'`)```bash

4. Push to the branch (`git push origin feature/new-feature`)docker-compose up

5. Create a Pull Request```



## 📜 License### 3. Docker (Production with Nginx)

```bash

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.docker-compose --profile production up

```

## 🙏 Acknowledgments

### 4. Cloud Deployment

- **BreakHis Dataset**: Spanhol, F. A., et al. "A dataset for breast cancer histopathological image classification." IEEE Transactions on Biomedical Engineering 63.7 (2015): 1455-1462.- Deploy to AWS, Google Cloud, Azure, or Heroku

- **ResNet Architecture**: He, K., et al. "Deep residual learning for image recognition." CVPR 2016.- Use the Docker image for containerized deployment

- **TensorFlow/Keras**: For providing the deep learning framework- Configure environment variables as needed



## ⚠️ Medical Disclaimer## Contributing



This application is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Medical Disclaimer

**IMPORTANT**: This application is designed for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical decisions. Do not delay seeking medical advice based on these results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## Acknowledgments

- BreakHis dataset creators and contributors
- TensorFlow/Keras development team
- Flask and Bootstrap communities
- Medical imaging research community
