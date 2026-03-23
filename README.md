# VisNet - Visual Speech Recognition Web Application

VisNet is a web application that performs visual speech recognition using deep learning. Upload a video of someone speaking, and the application will transcribe the speech by analysing lip movements.

## Live Demo    

Try the live demo on HuggingFace Spaces: [🤗 VisNet Demo](https://huggingface.co/spaces/okregent/visnet)

## Setup Instructions

### 1. Environment Setup

This project requires Python 3.9. If you don't have it installed, you can use pyenv to manage Python versions:

1. Install pyenv (macOS):
```bash
brew install pyenv
```

2. Install Git LFS (required for model files):
```bash
brew install git-lfs
git lfs install
```

3. Install Python 3.9:
```bash
pyenv install 3.9.10
```

4. Create and activate a virtual environment:

Create virtual environment with Python 3.9
```bash
~/.pyenv/versions/3.9.10/bin/python -m venv venv
```

Activate on Windows
```bash
venv\Scripts\activate
```

OR on Unix/MacOS
```bash
source venv/bin/activate
```

Install required packages
```bash
pip install -r requirements.txt
```

### 2. download the model and auto_avsr framework

The VisNet web app uses the auto_avsr framework to power model our model inference.

1. Clone the auto_avsr GitHub repository:

```bash
git clone https://github.com/mpc001/auto_avsr
```

2. Download the VisNet model from [here](https://drive.google.com/file/d/1C1gzm1Gn02AVsPN-ERl3gBqNWcAR7CMd/view?usp=drive_link) and place it in the root directory. Alternatively, you can download a model from the auto_avsr [model zoo](https://github.com/mpc001/auto_avsr?tab=readme-ov-file#model-zoo).

### 2. Face Detection Setup

The face detection components need to be installed from GitHub repositories:

1. First, ensure you have Git LFS installed

2. Clone and install face detection
```bash
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
```

3. Clone and install face alignment
```bash
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```

### 3. File Structure Setup

Create an uploads folder

```bash
mkdir uploads
```

Ensure you have the following directory structure:

```
root/
├── auto_avsr
├── face_alignment
├── face_detection
├── templates/         # HTML templates
│   ├── index.html
│   ├── upload.html
│   └── about.html
├── static/            # Static files
│   ├── style.css
│   ├── upload-script.js
│   ├── VisNet.png
│   └── file-icon.png
├── uploads/          # Temporary upload directory
├── app.py            # Main Flask application
├── config.py         # Config file
└── requirements.txt  # Python dependencies
```

### 4. Model Setup
1. Download the required model files:
   - Main model: Place the modle in the root directory and set the DEFAULT_MODEL_PATH
   - Face detection model: Uses ResNet50 and will be downloaded automatically during face_detection setup.
     If the model isn't loaded, download the ResNet50 elsewhere.
   - Face alignment model: will be downloaded during face_alignment setup

### 5. Running the Application

Make sure your virtual environment is activated

Windows
```bash
venv\Scripts\activate
```

Unix/MacOS
```bash
source venv/bin/activate
```

Start the Flask server
```bash
python app.py
```

The server will start at `http://localhost:5001`

## Usage Guide
1. Open `http://localhost:5001` in your web browser
2. Click "Try It" or navigate to the upload page
3. Either drag & drop a video file or click to browse
4. Supported formats: .mp4 (max size: 100MB)
5. Wait for processing to complete
6. View the transcription results
7. Use the copy or download buttons to save the results

## Supported File Types

- MP4 (.mp4)

## Technical Details

- The application uses Flask for the web server
- Frontend is built with HTML, JavaScript, and Tailwind CSS
- Video processing uses the retinaface detector for face detection
- The model uses a conformer-based architecture for visual speech recognition

## Notes

- Maximum file size is limited to 100MB
- Processing time depends on video length and hardware capabilities
- For optimal results, ensure good lighting and clear face visibility in videos

