import os
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from model_utils import BreastCancerClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize the classifier
try:
    # Try to load a pre-trained model if it exists
    model_path = 'models/breast_cancer_model.keras'
    classifier = BreastCancerClassifier(model_path if os.path.exists(model_path) else None)
    logger.info("Classifier initialized successfully")
except Exception as e:
    logger.error(f"Error initializing classifier: {e}")
    classifier = None

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and prediction"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Generate a unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Save the file
                file.save(filepath)
                logger.info(f"File saved: {filepath}")
                
                # Make prediction
                if classifier:
                    result = classifier.predict(filepath)
                    
                    if result:
                        logger.info(f"Prediction result: {result}")
                        return render_template('result.html', 
                                               result=result, 
                                               image_path=f"uploads/{unique_filename}")
                    else:
                        flash('Error processing the image. Please try again.')
                        # Clean up the uploaded file
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        return redirect(url_for('upload_file'))
                else:
                    flash('Model not available. Please try again later.')
                    # Clean up the uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return redirect(url_for('upload_file'))
                    
            except Exception as e:
                logger.error(f"Error processing upload: {e}")
                flash('Error processing the file. Please try again.')
                return redirect(url_for('upload_file'))
        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF).')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(filepath)
        
        # Make prediction
        if classifier:
            result = classifier.predict(filepath)
            
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if result:
                return jsonify(result)
            else:
                return jsonify({'error': 'Error processing the image'}), 500
        else:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Model not available'}), 503
            
    except Exception as e:
        logger.error(f"Error in API prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('upload_file'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
