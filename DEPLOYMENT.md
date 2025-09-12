# Deployment Instructions for Breast Cancer Classification App

## Quick Start

### Option 1: Local Development
```bash
# Navigate to the app directory
cd "c:\Users\Ephraim\Desktop\Breast Cancer App\breast-cancer-app"

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Option 2: Docker Deployment
```bash
# Navigate to the app directory
cd "c:\Users\Ephraim\Desktop\Breast Cancer App\breast-cancer-app"

# Build and run with Docker
docker-compose up --build
```

## Detailed Setup Guide

### 1. Environment Setup

**Prerequisites:**
- Python 3.8 or higher
- pip (Python package manager)
- Docker (optional, for containerized deployment)

### 2. Install Python Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install required packages
pip install -r requirements.txt
```

### 3. Model Setup

If you have a trained model:
1. Place your trained model file (`.keras` or `.h5`) in the `models/` directory
2. Update the model path in `app.py` if necessary

If you don't have a trained model:
- The app will create a new model architecture
- You can train it using your notebooks or use the demo mode

### 4. Running the Application

**Development Mode:**
```bash
python app.py
```
- App will be available at `http://localhost:5000`
- Debug mode enabled
- Auto-reload on code changes

**Production Mode with Gunicorn:**
```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
```

### 5. Docker Deployment

**Simple Docker Run:**
```bash
# Build the image
docker build -t breast-cancer-app .

# Run the container
docker run -p 5000:5000 breast-cancer-app
```

**Docker Compose (Recommended):**
```bash
# Development
docker-compose up --build

# Production (with Nginx)
docker-compose --profile production up --build
```

## Cloud Deployment Options

### Heroku Deployment

1. **Install Heroku CLI**
2. **Create Heroku app:**
```bash
heroku create your-app-name
```

3. **Add Procfile:**
```
web: gunicorn app:app
```

4. **Deploy:**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### AWS Deployment (EC2)

1. **Launch EC2 instance** (Ubuntu/Amazon Linux)
2. **Install dependencies:**
```bash
sudo apt update
sudo apt install python3-pip docker.io docker-compose
```

3. **Clone and deploy:**
```bash
git clone <your-repo>
cd breast-cancer-app
sudo docker-compose up --build -d
```

### Google Cloud Run

1. **Build and push to Container Registry:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/breast-cancer-app
```

2. **Deploy to Cloud Run:**
```bash
gcloud run deploy --image gcr.io/PROJECT_ID/breast-cancer-app --platform managed
```

## Configuration

### Environment Variables

Create a `.env` file (optional):
```
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
MAX_CONTENT_LENGTH=16777216
```

### Custom Configuration

Edit `app.py` to modify:
- Upload folder path
- Maximum file size
- Allowed file extensions
- Model parameters

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

2. **Model Loading Issues:**
   - Verify model file exists in `models/` directory
   - Check TensorFlow version compatibility

3. **Permission Issues (Docker):**
   - Add user to docker group: `sudo usermod -aG docker $USER`
   - Restart terminal or reboot

4. **Port Already in Use:**
   - Change port in `app.py`: `app.run(port=5001)`
   - Or kill process using port 5000

### Performance Optimization

1. **For Production:**
   - Use Gunicorn with multiple workers
   - Set up Nginx reverse proxy
   - Enable gzip compression
   - Use Redis for caching (optional)

2. **For Better Model Performance:**
   - Use GPU-enabled Docker image if available
   - Optimize model size and architecture
   - Implement model caching

## Security Considerations

### Production Security

1. **Set strong secret key:**
```python
app.secret_key = 'your-very-secure-secret-key'
```

2. **Use HTTPS:**
   - Configure SSL certificates
   - Use reverse proxy (Nginx)

3. **File Upload Security:**
   - Validate file types and sizes
   - Scan for malware (optional)
   - Implement rate limiting

### Privacy Protection

- Uploaded images are automatically deleted after processing
- No personal data is stored
- Implement proper logging and monitoring

## Monitoring and Maintenance

### Health Checks

The app includes health check endpoints:
- Basic health: `GET /`
- Application status monitoring

### Logging

Add logging configuration:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Backup and Recovery

- Regular backups of model files
- Database backups (if using persistent storage)
- Configuration file backups

## Scaling Considerations

### Horizontal Scaling

1. **Load Balancer Setup:**
   - Use Nginx or cloud load balancers
   - Multiple app instances

2. **Container Orchestration:**
   - Kubernetes deployment
   - Docker Swarm mode

### Database Scaling

If adding database functionality:
- Use managed database services
- Implement connection pooling
- Consider read replicas

## Support and Maintenance

### Regular Updates

1. **Dependencies:**
```bash
pip list --outdated
pip install -U package_name
```

2. **Security patches:**
   - Monitor security advisories
   - Update base Docker images

3. **Model updates:**
   - Retrain with new data
   - A/B test model versions

### Monitoring

Set up monitoring for:
- Application performance
- Error rates
- Resource usage
- User activity

For questions or issues, refer to the main README.md or create an issue in the repository.
