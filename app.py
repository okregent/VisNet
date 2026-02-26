import os
import sys
import torch
import argparse
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import DEFAULT_MODEL_PATH

# Add the auto_avsr directory to Python path
sys.path.insert(0, "./auto_avsr")

from lightning import ModelModule
from datamodule.transforms import VideoTransform
from preparation.detectors.retinaface.detector import LandmarksDetector
from preparation.detectors.retinaface.video_process import VideoProcess

app = Flask(__name__, static_url_path='', template_folder='templates')
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Initialize model and pipeline components
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args(args=[])
setattr(args, 'modality', 'video')

class InferencePipeline(torch.nn.Module):
    def __init__(self, args, ckpt_path, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = args.modality
        
        # Initialize video components
        self.landmarks_detector = LandmarksDetector(device="cuda:0" if torch.cuda.is_available() else "cpu")
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

        # Load model
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.modelmodule = ModelModule(args)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()

    # Duration of each video segment in seconds.
    # The model was trained on short clips (LRS3 dataset, ~1-16s),
    # so splitting long videos into segments improves accuracy.
    SEGMENT_DURATION = 5

    # Minimum number of frames required to run inference on a segment.
    # Segments shorter than this are skipped to avoid poor predictions.
    MIN_FRAMES = 10

    def load_video(self, data_filename):
        """
        Load a video file and return its frames as a numpy array along with FPS.

        torchvision.io.read_video returns a tuple of:
          - video frames tensor: shape (T, H, W, C)
          - audio frames tensor
          - metadata dict containing 'video_fps'

        We convert the video tensor to numpy for downstream processing.
        """
        import torchvision
        frames, _, info = torchvision.io.read_video(data_filename, pts_unit="sec")
        # Extract FPS from metadata; fall back to 25 if not available
        fps = info.get("video_fps", 25.0)
        return frames.numpy(), fps

    def _process_segment(self, segment_frames):
        """
        Run the full preprocessing and inference pipeline on a single segment.

        Steps:
          1. Detect facial landmarks using RetinaFace
          2. Crop the lip region based on landmarks
          3. Convert to tensor and rearrange dimensions
          4. Normalize and apply test-time transforms
          5. Run the Conformer model to produce a transcript

        Returns an empty string if face detection fails for this segment.
        """
        # Step 1: Detect facial landmarks for every frame in the segment.
        # landmarks is a list of landmark arrays, one per frame.
        landmarks = self.landmarks_detector(segment_frames)

        # Step 2: Crop the lip region from each frame using the detected landmarks.
        # Returns None if no face is detected in the segment.
        processed = self.video_process(segment_frames, landmarks)
        if processed is None:
            return ""

        # Step 3: Convert numpy array to PyTorch tensor.
        # Shape after conversion: (T, H, W, C) where T=frames, C=channels
        video_tensor = torch.tensor(processed)

        # Step 4: Rearrange from (T, H, W, C) to (T, C, H, W).
        # PyTorch's convolutional layers expect channels before spatial dims.
        video_tensor = video_tensor.permute((0, 3, 1, 2))

        # Step 5: Apply normalization and center crop (defined in VideoTransform).
        video_tensor = self.video_transform(video_tensor)

        # Step 6: Run inference without computing gradients.
        # torch.no_grad() saves memory and speeds up inference
        # since we don't need backpropagation at prediction time.
        with torch.no_grad():
            transcript = self.modelmodule(video_tensor)

        return transcript.strip()

    def forward(self, data_filename):
        """
        Full inference pipeline for a video file.

        To handle videos longer than what the model was trained on,
        we split the video into fixed-length segments, run inference
        on each segment independently, and join the results.
        """
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        # Load all video frames and the actual FPS of the video
        video, fps = self.load_video(data_filename)

        # Calculate how many frames correspond to SEGMENT_DURATION seconds
        segment_size = int(fps * self.SEGMENT_DURATION)

        total_frames = len(video)
        transcripts = []

        # Slide through the video in steps of segment_size
        for start in range(0, total_frames, segment_size):
            end = min(start + segment_size, total_frames)
            segment = video[start:end]

            # Skip segments that are too short for reliable inference
            if len(segment) < self.MIN_FRAMES:
                continue

            # Run inference on this segment and collect the result
            result = self._process_segment(segment)
            if result:
                transcripts.append(result)

        # Join all segment transcripts with a space between them
        return " ".join(transcripts)

# Initialize the pipeline
pipeline = InferencePipeline(args, DEFAULT_MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

# Route for the home page
@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run inference
            transcript = pipeline(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({'transcript': transcript})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Move HTML files to templates if they exist in root
    for html_file in ['index.html', 'upload.html', 'about.html']:
        if os.path.exists(html_file) and not os.path.exists(os.path.join('templates', html_file)):
            os.rename(html_file, os.path.join('templates', html_file))
    
    # Move static files to static directory if they exist in root
    for static_file in ['style.css', 'upload-script.js', 'VisNet.png', 'file-icon.png']:
        if os.path.exists(static_file) and not os.path.exists(os.path.join('static', static_file)):
            os.rename(static_file, os.path.join('static', static_file))
    
    print("Server is running at http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)