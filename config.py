"""
Configuration file for Football Analysis Project
Contains all constants, file paths, and model configurations
"""

import os
from typing import Dict, List

# ============================================================================
# FILE PATHS AND DIRECTORIES
# ============================================================================

# Video paths (modify these according to your setup)
SOURCE_VIDEO_PATH = "/content/drive/MyDrive/1223.mp4"
OUTPUT_VIDEO_PATH = "/content/output.mp4"
SHORTENED_VIDEO_PATH = "/content/drive/MyDrive/shortened_video_30s.mp4"

# Model checkpoints
SAM_CHECKPOINT_PATH = '/content/drive/MyDrive/checkpoint.pt'
SAM_MODEL_PATH = 'sam2.1_b.pt'

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Roboflow API and Model IDs
ROBOFLOW_API_KEY_ENV = 'ROBOFLOW_API_KEY'  # Environment variable name
PLAYER_DETECTION_MODEL_ID = "label_football/6"
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/18"  # Add field detection model ID

# ============================================================================
# DETECTION CLASSES
# ============================================================================

# Class IDs for different objects in the detection model
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

# Detection and tracking parameters
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
TRACKER_BUFFER = 30
MIN_PLAYER_SIZE = 1000  # Minimum area to filter noise

# Video processing parameters
STRIDE = 30  # Frame stride for team classification training
BATCH_SIZE = 16  # Batch size for model inference (when implementing batch processing)

# Team classification parameters
TEAM_CLASSIFICATION_INTERVAL = 30  # Re-classify team every N frames
TEAM_HISTORY_LENGTH = 5  # Number of previous predictions to consider for smoothing

# Ball control calculation
BALL_CONTROL_DISTANCE_THRESHOLD = 50  # Distance threshold for ball possession

# ============================================================================
# VISUALIZATION COLORS
# ============================================================================

# Color scheme for different teams and objects
COLORS = {
    # Player colors by team
    'team_0_players': '#00BFFF',      # Blue for team 0 players
    'team_1_players': '#FF1493',      # Pink for team 1 players
    
    # Goalkeeper colors by team
    'team_0_goalkeeper': '#32CD32',   # Green for team 0 goalkeeper
    'team_1_goalkeeper': '#FF0000',   # Red for team 1 goalkeeper
    
    # Other objects
    'referee': '#000000',             # Black for referees
    'ball': '#FFD700',                # Gold for ball
    
    # Text colors
    'text_light': '#FFFFFF',          # White text
    'text_dark': '#000000',           # Black text
}

# ============================================================================
# ANNOTATOR SETTINGS
# ============================================================================

# Ellipse annotator settings
ELLIPSE_THICKNESS = 2

# Label annotator settings
LABEL_TEXT_POSITION = "BOTTOM_CENTER"

# Triangle annotator settings (for ball)
TRIANGLE_BASE = 25
TRIANGLE_HEIGHT = 21
TRIANGLE_OUTLINE_THICKNESS = 1

# ============================================================================
# PITCH ANALYSIS SETTINGS
# ============================================================================

# Pitch dimensions (in meters, standard football pitch)
PITCH_LENGTH = 105  # meters
PITCH_WIDTH = 68   # meters

# Voronoi diagram settings
VORONOI_COLORS = {
    'team_0': '#00BFFF',
    'team_1': '#FF1493',
    'alpha': 0.3  # Transparency for Voronoi regions
}

# ============================================================================
# DEVICE SETTINGS
# ============================================================================

# Device configuration
DEVICE = "cuda"  # Use "cpu" if CUDA is not available

# ONNX Runtime settings
ONNX_EXECUTION_PROVIDERS = "[CUDAExecutionProvider]"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_color_by_role_and_team(class_id: int, team_id: int = None) -> str:
    """
    Get color hex code based on object class and team ID
    
    Args:
        class_id: Object class ID (BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID)
        team_id: Team ID (0 or 1), only relevant for players and goalkeepers
    
    Returns:
        Hex color code string
    """
    if class_id == BALL_ID:
        return COLORS['ball']
    elif class_id == REFEREE_ID:
        return COLORS['referee']
    elif class_id == GOALKEEPER_ID:
        if team_id == 0:
            return COLORS['team_0_goalkeeper']
        elif team_id == 1:
            return COLORS['team_1_goalkeeper']
        else:
            return COLORS['text_light']  # Default color
    elif class_id == PLAYER_ID:
        if team_id == 0:
            return COLORS['team_0_players']
        elif team_id == 1:
            return COLORS['team_1_players']
        else:
            return COLORS['text_light']  # Default color
    else:
        return COLORS['text_light']  # Default color

def validate_config():
    """
    Validate configuration settings and check for required environment variables
    """
    # Check if required environment variables are set
    if ROBOFLOW_API_KEY_ENV not in os.environ:
        print(f"Warning: {ROBOFLOW_API_KEY_ENV} environment variable not set")
    
    # Validate file paths exist (skip if using Google Colab paths)
    if not SOURCE_VIDEO_PATH.startswith("/content/"):
        if not os.path.exists(SOURCE_VIDEO_PATH):
            print(f"Warning: Source video path does not exist: {SOURCE_VIDEO_PATH}")
    
    print("Configuration loaded successfully!")

# Run validation when module is imported
if __name__ == "__main__":
    validate_config() 