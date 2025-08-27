# AI Facial Expression Transformer

A web application for AI-powered facial expression modification using deep learning with upload interface, conversion history, and REST API.

## 🚀 Features

- **Drag-and-drop image upload** with preview and validation
- **AI-powered facial expression transformation** using diffusion models
- **Real-time processing status** and result preview
- **Conversion history** with thumbnail gallery and metadata
- **RESTful API** with authentication for external integration
- **Batch processing capabilities** for multiple images
- **Image optimization** and format conversion (JPEG/PNG/WebP)
- **User session management** and temporary file cleanup

## 🛠️ Tech Stack

- **Backend**: Python, Flask, SQLAlchemy
- **AI/ML**: PyTorch, Diffusers, Transformers, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Database**: SQLite (development), PostgreSQL (production)
- **Authentication**: JWT tokens
- **Caching**: Redis (optional)

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended for AI models)
- GPU support (optional but recommended for faster inference)

## 🔧 Installation

### 1. Clone and Setup Environment

```bash
# Create project directory
mkdir facial_expression_app
cd facial_expression_app

# Create virtual environment
python -m venv facial_expression_env
source facial_expression_env/bin/activate  # On Windows: facial_expression_env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Database Setup

```bash
# Initialize database
python database/migrations.py
```

### 4. Configuration

Create a `.env` file in the root directory:

```env
# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DEBUG=True

# Database Configuration
DATABASE_URL=sqlite:///facial_expression.db

# AI Model Configuration
MODEL_CACHE_DIR=./model_cache
MAX_IMAGE_SIZE=2048
PROCESSING_TIMEOUT=300

# API Configuration
JWT_SECRET_KEY=your-jwt-secret-key
API_RATE_LIMIT=100

# File Upload Configuration
UPLOAD_FOLDER=./static/uploads
MAX_CONTENT_LENGTH=16777216  # 16MB

# Optional: Redis Configuration
REDIS_URL=redis://localhost:6379/0
```

## 🚀 Running the Application

### Development Mode

```bash
# Start the Flask development server
python app.py

# Application will be available at:
# Web Interface: http://localhost:5000
# API Documentation: http://localhost:5000/api/health
```

### Production Mode

```bash
# Set production environment
export FLASK_ENV=production

# Run with Gunicorn (install separately)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📖 Usage Guide

### Web Interface

1. **Upload Image**: 
   - Visit `http://localhost:5000`
   - Drag and drop an image or click to browse
   - Supported formats: JPEG, PNG, WebP

2. **Select Expression**:
   - Choose target expression: Happy, Sad, Angry, Surprised, Neutral
   - Adjust intensity slider (0.1 - 1.0)

3. **Process Image**:
   - Click "Transform Expression"
   - Monitor real-time processing status
   - Download result when complete

4. **View History**:
   - Access conversion history at `/history`
   - Search and filter past conversions
   - Download or delete previous results

### REST API

#### Authentication

```bash
# Get API token
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "api_user", "password": "your_password"}'
```

#### Upload and Convert

```bash
# Upload image for conversion
curl -X POST http://localhost:5000/api/convert \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "image=@path/to/image.jpg" \
  -F "target_expression=happy" \
  -F "intensity=0.8"
```

#### Check Status

```bash
# Check conversion status
curl -X GET http://localhost:5000/api/status/CONVERSION_ID \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### Get History

```bash
# Get conversion history
curl -X GET http://localhost:5000/api/history \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 🏗️ Project Structure

```
facial_expression_app/
├── app.py                          # Flask main application
├── config.py                       # App configuration
├── requirements.txt                # Dependencies
├── models/
│   ├── __init__.py
│   ├── expression_model.py         # Deep learning model wrapper
│   └── diffusion_pipeline.py      # Diffusion model implementation
├── api/
│   ├── __init__.py
│   ├── routes.py                   # REST API endpoints
│   └── auth.py                     # API authentication
├── services/
│   ├── __init__.py
│   ├── image_processor.py          # Image preprocessing/postprocessing
│   ├── model_service.py            # Model inference service
│   └── storage_service.py          # File and database operations
├── database/
│   ├── __init__.py
│   ├── models.py                   # SQLAlchemy database models
│   └── migrations.py               # Database setup
├── static/
│   ├── css/style.css               # Frontend styling
│   ├── js/main.js                  # Frontend JavaScript
│   └── uploads/                    # Temporary upload directory
├── templates/
│   ├── index.html                  # Main upload interface
│   ├── results.html                # Results display
│   └── history.html                # Conversion history
└── README.md                       # Documentation
```

## 🔧 API Endpoints

### Core Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Main web interface | No |
| GET | `/history` | Conversion history page | No |
| GET | `/results/<id>` | View conversion results | No |
| GET | `/api/health` | API health check | No |

### Authentication

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/auth/login` | Get JWT token | No |
| POST | `/api/auth/refresh` | Refresh JWT token | Yes |

### Image Processing

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/convert` | Start image conversion | Yes |
| GET | `/api/status/<id>` | Check conversion status | Yes |
| GET | `/api/result/<id>` | Get conversion result | Yes |

### History Management

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/history` | Get conversion history | Yes |
| DELETE | `/api/history/<id>` | Delete specific conversion | Yes |
| DELETE | `/api/history` | Clear all history | Yes |
| GET | `/api/history/export` | Export history data | Yes |

## 🎨 Supported Expressions

- **Happy**: Smile, joy, positive emotions
- **Sad**: Sorrow, melancholy, downturned features
- **Angry**: Frown, tension, aggressive features
- **Surprised**: Wide eyes, raised eyebrows, open mouth
- **Neutral**: Relaxed, natural expression
- **Fear**: Worried, anxious, tense features
- **Disgust**: Wrinkled nose, negative reaction

## ⚙️ Configuration Options

### Environment Variables

- `FLASK_ENV`: Application environment (development/production)
- `SECRET_KEY`: Flask secret key for sessions
- `DATABASE_URL`: Database connection string
- `MODEL_CACHE_DIR`: Directory for cached AI models
- `MAX_IMAGE_SIZE`: Maximum image dimension (pixels)
- `PROCESSING_TIMEOUT`: Maximum processing time (seconds)
- `JWT_SECRET_KEY`: Secret key for JWT tokens
- `API_RATE_LIMIT`: API requests per minute
- `REDIS_URL`: Redis connection string (optional)

### Model Configuration

- **Default Model**: Stable Diffusion with ControlNet
- **Fallback Model**: OpenCV-based expression transfer
- **Cache Strategy**: Automatic model caching and loading
- **GPU Acceleration**: Automatic detection and usage

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   ```bash
   # Reduce image size or use CPU-only mode
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Model Download Issues**:
   ```bash
   # Clear model cache and retry
   rm -rf ./model_cache
   python -c "from models.expression_model import ExpressionModel; ExpressionModel().load_model()"
   ```

3. **Database Connection Errors**:
   ```bash
   # Reset database
   rm facial_expression.db
   python database/migrations.py
   ```

4. **Port Already in Use**:
   ```bash
   # Use different port
   export FLASK_RUN_PORT=5001
   python app.py
   ```

### Performance Optimization

1. **Enable GPU Acceleration**:
   - Install CUDA-compatible PyTorch
   - Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Use Redis Caching**:
   - Install and start Redis server
   - Set `REDIS_URL` in environment

3. **Optimize Image Processing**:
   - Reduce `MAX_IMAGE_SIZE` for faster processing
   - Enable image compression in config

## 📊 Monitoring and Logging

### Health Checks

- **Application Health**: `GET /api/health`
- **Database Status**: Included in health check
- **Model Status**: Automatic model loading verification

### Logging

- **Development**: Console output with DEBUG level
- **Production**: File-based logging with INFO level
- **Error Tracking**: Automatic error logging and reporting

## 🔒 Security Considerations

- **File Upload Validation**: Strict file type and size checking
- **JWT Authentication**: Secure API access with token expiration
- **Input Sanitization**: Protection against malicious uploads
- **Rate Limiting**: API request throttling
- **CORS Configuration**: Controlled cross-origin access

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Cloud Deployment

- **Heroku**: Use provided `Procfile` and environment variables
- **AWS**: Deploy with Elastic Beanstalk or ECS
- **Google Cloud**: Use App Engine or Cloud Run
- **Azure**: Deploy with App Service

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **Issues**: GitHub Issues tracker
- **Documentation**: This README and inline code comments
- **Community**: Discussions tab for questions and feedback

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
  - Web interface for image upload and conversion
  - REST API with authentication
  - Conversion history and management
  - AI-powered expression transformation

---

**Built with ❤️ using Flask, PyTorch, and modern web technologies**