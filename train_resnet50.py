import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom F1 Score metric (from the notebook)
def f1_score(y_true, y_pred):
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

# ResNet50 Model (exact same as in notebook)
def build_resnet50_model(input_shape=(128, 128, 3), num_classes=2, learning_rate=1e-5, dropout_rate=0.6):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.02))
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, epsilon=1e-07),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='AUC'), f1_score]
    )

    return model

def create_data_generators():
    """Create data generators for training and validation"""
    # Data augmentation (same as in notebook)
    data_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data generator (no augmentation)
    val_data_gen = ImageDataGenerator(rescale=1./255)
    
    return data_gen, val_data_gen

def prepare_sample_data():
    """Create a small sample dataset for demonstration"""
    # Create sample images (random data for demonstration)
    # In practice, you'd load your actual BreakHis dataset here
    num_samples = 100
    X_sample = np.random.rand(num_samples, 128, 128, 3) * 255
    y_sample = np.random.randint(0, 2, num_samples)
    
    # Convert to categorical
    y_sample_cat = to_categorical(y_sample, 2)
    
    # Split into train/val
    split_idx = int(0.8 * num_samples)
    X_train = X_sample[:split_idx]
    y_train = y_sample_cat[:split_idx]
    X_val = X_sample[split_idx:]
    y_val = y_sample_cat[split_idx:]
    
    return X_train, y_train, X_val, y_val

def train_resnet50_model():
    """Train the ResNet50 model"""
    logger.info("Starting ResNet50 model training...")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        # Enable memory growth to avoid taking all GPU memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPU found. Using CPU.")
    
    # Build model
    model = build_resnet50_model()
    logger.info("ResNet50 model created successfully")
    
    # Prepare data (in practice, load your actual BreakHis dataset)
    X_train, y_train, X_val, y_val = prepare_sample_data()
    logger.info(f"Sample data prepared: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    
    # Create data generators
    data_gen, val_data_gen = create_data_generators()
    
    # Callbacks (same as in notebook)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-12, verbose=1)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    
    callbacks = [earlystop, reduce_lr, checkpoint]
    
    # Training parameters
    epochs = 50  # Reduced for demo
    batch_size = 32
    
    logger.info("Starting training...")
    
    # Train the model
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
        callbacks=callbacks,
        epochs=epochs,
        steps_per_epoch=X_train.shape[0] // batch_size,
        validation_data=val_data_gen.flow(X_val, y_val, batch_size=batch_size),
        validation_steps=X_val.shape[0] // batch_size,
        verbose=1
    )
    
    # Save the final model
    model.save("resnet50_breast_cancer.keras")
    logger.info("Model saved as resnet50_breast_cancer.keras")
    
    # Save training history
    import pickle
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    logger.info("Training completed!")
    return model, history

if __name__ == "__main__":
    model, history = train_resnet50_model()
    
    # Print final metrics
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nFinal Results:")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    print(f"Validation Loss: {final_val_loss:.4f}")
