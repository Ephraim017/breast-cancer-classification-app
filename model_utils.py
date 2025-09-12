import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
import cv2
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup GPU if available"""
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            logger.info(f"Found {len(physical_devices)} GPU(s). Enabling memory growth.")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            return True
        else:
            logger.info("No GPU found. Using CPU.")
            return False
    except Exception as e:
        logger.warning(f"GPU setup failed: {e}. Using CPU.")
        return False

# Setup GPU
gpu_available = setup_gpu()

# Custom F1 Score metric (for model compatibility)
def f1_score_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall_val = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall_val

    def precision(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision_val = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision_val

    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2*((precision_val*recall_val)/(precision_val+recall_val+tf.keras.backend.epsilon()))

class BreastCancerClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.size = 128  # Changed to 128 to match notebook
        self.input_shape = (128, 128, 3)  # Match notebook input shape
        self.num_classes = 2  # Benign, Malignant
        self.class_names = ['Benign', 'Malignant']
        
        # Try to load trained model first
        trained_model_paths = [
            'resnet50_breast_cancer_final.keras',
            'best_resnet50_model.keras',
            'resnet50_model.keras',
            model_path
        ]
        
        model_loaded = False
        for path in trained_model_paths:
            if path and os.path.exists(path):
                try:
                    self.load_model(path)
                    model_loaded = True
                    logger.info(f"Successfully loaded trained model from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {e}")
        
        if not model_loaded:
            logger.info("No pre-trained model found. Creating new model architecture.")
            logger.warning("Note: For accurate predictions, please train the model on your dataset first.")
            self.create_model()
    
    def create_model(self):
        """Create ResNet50-based model architecture (matching notebook)"""
        try:
            # Use ResNet50 with ImageNet weights (same as notebook)
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Make all layers trainable (same as notebook)
            base_model.trainable = True
            
            # Build model with Sequential API (same as notebook)
            self.model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
                Dropout(0.6),  # Same dropout rate as notebook
                Dense(self.num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.02))
            ])
            
            # Compile with same settings as notebook
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-07),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("ResNet50-based model created successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error creating ResNet50 model: {e}")
            # Fallback to simple CNN
            return self.create_simple_model()
    
    def create_simple_model(self):
        """Fallback simple CNN model for breast cancer classification"""
        logger.info("Creating fallback CNN model")
        
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.size, self.size, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            # Load model with custom objects for compatibility
            custom_objects = {'f1_score_metric': f1_score_metric}
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating new model...")
            self.create_model()
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction (matching training preprocessing)"""
        try:
            # Load image using PIL
            img = Image.open(image_path)
            img = img.convert('RGB')
            
            # Resize to model input size
            img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Simple normalization to [0, 1] - SAME AS TRAINING
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """Make prediction on a single image with enhanced confidence scoring"""
        if self.model is None:
            logger.error("Model not initialized")
            return None
        
        # Preprocess the image
        processed_image = self.preprocess_image(image_path)
        
        if processed_image is None:
            return None
        
        try:
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # DEBUG: Log raw predictions
            logger.info(f"DEBUG: Raw model output: {predictions}")
            logger.info(f"DEBUG: Output shape: {predictions.shape}")
            
            # For categorical classification with softmax activation
            # predictions shape: (1, 2) where [benign_prob, malignant_prob]
            benign_prob = float(predictions[0][0])
            malignant_prob = float(predictions[0][1])
            
            logger.info(f"DEBUG: Benign prob (index 0): {benign_prob}")
            logger.info(f"DEBUG: Malignant prob (index 1): {malignant_prob}")
            
            # Determine predicted class
            if malignant_prob > benign_prob:
                predicted_class = 'Malignant'
                confidence = malignant_prob
            else:
                predicted_class = 'Benign'
                confidence = benign_prob
            
            logger.info(f"DEBUG: Comparison - malignant_prob ({malignant_prob}) > benign_prob ({benign_prob}): {malignant_prob > benign_prob}")
            logger.info(f"DEBUG: Final prediction: {predicted_class}")
            
            # Enhanced risk assessment
            risk_level = self.get_enhanced_risk_level(malignant_prob)
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'benign_probability': benign_prob,
                'malignant_probability': malignant_prob,
                'risk_level': risk_level,
                'interpretation': self.get_interpretation(malignant_prob)
            }
            
            logger.info(f"Prediction: {predicted_class} ({confidence:.2%} confidence)")
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def get_enhanced_risk_level(self, malignant_prob):
        """Enhanced risk level assessment"""
        if malignant_prob < 0.2:
            return "Very Low Risk"
        elif malignant_prob < 0.4:
            return "Low Risk"
        elif malignant_prob < 0.6:
            return "Moderate Risk"
        elif malignant_prob < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def get_interpretation(self, malignant_prob):
        """Provide clinical interpretation"""
        if malignant_prob < 0.3:
            return "Image characteristics suggest benign tissue with low likelihood of malignancy."
        elif malignant_prob < 0.7:
            return "Image shows mixed characteristics. Further clinical evaluation recommended."
        else:
            return "Image characteristics suggest high likelihood of malignancy. Immediate clinical attention recommended."
    
    def save_model(self, save_path):
        """Save the current model"""
        if self.model:
            try:
                self.model.save(save_path)
                logger.info(f"Model saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
    
    def train_model(self, train_data, validation_data, epochs=50):
        """Train the model on breast cancer data"""
        if self.model is None:
            self.create_model()
        
        # Callbacks for training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        try:
            history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return None

# Demo model for testing
def create_demo_model():
    """Create a demo model that gives reasonable predictions for testing"""
    logger.info("Creating demo model for testing purposes...")
    
    # Simple model that uses basic image statistics for prediction
    class DemoClassifier:
        def __init__(self):
            self.class_names = ['Benign', 'Malignant']
        
        def predict(self, image_path):
            try:
                # Load and analyze image
                img = Image.open(image_path).convert('RGB')
                img_array = np.array(img)
                
                # Simple heuristics based on image characteristics
                # (This is just for demo - real model needs training)
                avg_intensity = np.mean(img_array)
                contrast = np.std(img_array)
                
                # Simple rule-based prediction (for demo only)
                if avg_intensity > 150 and contrast > 40:
                    malignant_prob = 0.7  # Higher probability of malignant
                elif avg_intensity < 100 or contrast < 20:
                    malignant_prob = 0.2  # Lower probability
                else:
                    malignant_prob = 0.5  # Uncertain
                
                benign_prob = 1.0 - malignant_prob
                
                if malignant_prob > 0.5:
                    predicted_class = 'Malignant'
                    confidence = malignant_prob
                else:
                    predicted_class = 'Benign'
                    confidence = benign_prob
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'benign_probability': benign_prob,
                    'malignant_probability': malignant_prob,
                    'risk_level': 'Demo Mode',
                    'interpretation': 'This is a demo prediction. Please use a trained model for actual diagnosis.'
                }
                
            except Exception as e:
                logger.error(f"Demo prediction error: {e}")
                return None
    
    return DemoClassifier()
