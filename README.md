# Football Analysis Project

An AI-powered football video analysis system for player detection, tracking, and tactical analysis.

## Key Features

- **Object Detection**: Detect players, goalkeepers, referees, and ball in video footage
- **Object Tracking**: Maintain consistent IDs for players across video frames
- **Team Classification**: Automatically distinguish between two teams based on jersey colors
- **Ball Possession Analysis**: Calculate ball possession percentages for each team
- **Movement Statistics**: Track player movement distance and activity zones
- **Formation Analysis**: Monitor team positions and tactical patterns

## Project Structure

```
FootBall-Detection/
├── main.py                 # Main script to run analysis
├── config.py              # Configuration and constants
├── video_processor.py     # Video processing and object detection
├── analysis.py            # Ball possession and statistics analysis
├── requirements.txt       # Required dependencies
├── TODO.md               # Project improvement roadmap
├── README.md             # Documentation (this file)
└── Scout_Football.ipynb  # Original notebook (for reference)
```

## Installation

### 1. Install Basic Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Sports Library from Roboflow

```bash
pip install git+https://github.com/roboflow/sports.git
```

### 3. Setup API Key

You need a Roboflow API key to use the detection models:

#### On Google Colab:

```python
from google.colab import userdata
# Add ROBOFLOW_API_KEY to Colab Secrets
```

#### On Local Machine:

```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

## Usage

### Run Basic Analysis

```python
python main.py
```

### Customize Video Paths

Edit the `config.py` file:

```python
SOURCE_VIDEO_PATH = "path/to/your/video.mp4"
OUTPUT_VIDEO_PATH = "path/to/output/video.mp4"
```

### Using Individual Modules

#### Video Processing

```python
from video_processor import VideoProcessor
from config import *

# Initialize processor
processor = VideoProcessor(detection_model)

# Process video
processor.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    team_classifier=team_classifier
)
```

#### Ball Possession Analysis

```python
from analysis import BallPossessionAnalyzer

analyzer = BallPossessionAnalyzer()

# Update in frame processing loop
possession = analyzer.update_possession(
    ball_detections, player_detections, player_team_ids, frame_number
)

# Get final statistics
percentages = analyzer.get_possession_percentages(total_frames)
```

## Configuration

### Main Parameters in `config.py`

- `CONFIDENCE_THRESHOLD`: Confidence threshold for object detection (default: 0.3)
- `NMS_THRESHOLD`: Non-Maximum Suppression threshold (default: 0.5)
- `TEAM_CLASSIFICATION_INTERVAL`: Frame interval for team re-classification (default: 30)
- `BALL_CONTROL_DISTANCE_THRESHOLD`: Distance threshold for ball possession (default: 50)

### Colors

Team and object colors are defined in the `COLORS` dictionary:

```python
COLORS = {
    'team_0_players': '#00BFFF',      # Blue for team 0
    'team_1_players': '#FF1493',      # Pink for team 1
    'team_0_goalkeeper': '#32CD32',   # Green for team 0 goalkeeper
    'team_1_goalkeeper': '#FF0000',   # Red for team 1 goalkeeper
    'referee': '#000000',             # Black for referees
    'ball': '#FFD700',                # Gold for ball
}
```

## Output

The project generates:

1. **Annotated Video**: Video with bounding boxes, player IDs, and ball possession information
2. **Log File**: `football_analysis.log` containing detailed processing information
3. **Final Statistics**: Printed to console and log file

### Example Statistics Output:

```
Team 0 possession: 52.3%
Team 1 possession: 47.7%
Total possession switches: 23
```

## Performance Optimization

### GPU

The project is optimized for GPU usage. Ensure you have:

- CUDA installed
- PyTorch with CUDA support
- ONNX Runtime GPU

### Memory

For processing large videos:

- Reduce `BATCH_SIZE` in config if running out of GPU memory
- Increase `STRIDE` to process fewer frames during team classifier training

## Troubleshooting

### Common Issues

1. **"Roboflow API key not found"**

   - Check that API key is set correctly
   - Ensure you have access to the model

2. **"Could not import sports library"**

   ```bash
   pip install git+https://github.com/roboflow/sports.git
   ```

3. **CUDA errors**

   - Check CUDA compatibility
   - Set `DEVICE = "cpu"` in config.py if needed

4. **Memory errors**
   - Reduce batch size
   - Process shorter videos for testing

### Logging

Check the log file `football_analysis.log` for debugging:

```bash
tail -f football_analysis.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project uses open-source libraries. See requirements.txt for details.

## Contact

If you have any issues or questions, please create an issue on the GitHub repository.
