#!/usr/bin/env python3
"""
Main entry point for Football Analysis Project
Orchestrates the entire workflow from model loading to video analysis
"""

import os
import logging
from typing import Optional
from google.colab import userdata

# Import project modules
from config import *
from video_processor import VideoProcessor, collect_team_training_crops
from analysis import BallPossessionAnalyzer, PlayerMovementAnalyzer, FormationAnalyzer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('football_analysis.log')
        ]
    )

def setup_environment():
    """Setup environment variables and device configurations"""
    # Set ONNX Runtime execution providers
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = ONNX_EXECUTION_PROVIDERS
    
    # Validate configuration
    validate_config()
    
    logging.info("Environment setup completed")

def load_models():
    """
    Load all required models for the analysis
    
    Returns:
        Tuple of (detection_model, field_model, sam_model)
    """
    logging.info("Loading models...")
    
    # Load detection model
    try:
        from inference import get_model
        
        # Get API key from environment or userdata
        api_key = None
        if ROBOFLOW_API_KEY_ENV in os.environ:
            api_key = os.environ[ROBOFLOW_API_KEY_ENV]
        else:
            try:
                api_key = userdata.get(ROBOFLOW_API_KEY_ENV)
            except:
                logging.warning("Could not retrieve Roboflow API key from userdata")
        
        if not api_key:
            raise ValueError(f"Roboflow API key not found. Please set {ROBOFLOW_API_KEY_ENV}")
        
        # Load player detection model
        detection_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=api_key)
        logging.info(f"Player detection model loaded: {PLAYER_DETECTION_MODEL_ID}")
        
        # Load field detection model (optional)
        field_model = None
        try:
            field_model = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=api_key)
            logging.info(f"Field detection model loaded: {FIELD_DETECTION_MODEL_ID}")
        except Exception as e:
            logging.warning(f"Could not load field detection model: {e}")
        
    except ImportError:
        logging.error("Could not import inference library. Please install inference-gpu.")
        raise
    except Exception as e:
        logging.error(f"Error loading detection models: {e}")
        raise
    
    # Load SAM model (optional)
    sam_model = None
    try:
        from ultralytics import SAM
        import torch
        
        if os.path.exists(SAM_CHECKPOINT_PATH):
            sam_model = SAM(SAM_MODEL_PATH)
            checkpoint = torch.load(SAM_CHECKPOINT_PATH, map_location=DEVICE)
            
            if 'model' in checkpoint:
                sam_model.model.load_state_dict(checkpoint['model'])
                logging.info("SAM model loaded successfully")
            else:
                logging.warning("SAM checkpoint does not contain model weights")
        else:
            logging.warning(f"SAM checkpoint not found at {SAM_CHECKPOINT_PATH}")
    
    except Exception as e:
        logging.warning(f"Could not load SAM model: {e}")
    
    return detection_model, field_model, sam_model

def train_team_classifier(detection_model, video_path: str = SOURCE_VIDEO_PATH):
    """
    Train team classification model on the provided video
    
    Args:
        detection_model: Player detection model
        video_path: Path to training video
        
    Returns:
        Trained team classifier
    """
    logging.info("Training team classifier...")
    
    try:
        from sports.common.team import TeamClassifier
        
        # Collect training crops
        crops = collect_team_training_crops(video_path, detection_model, stride=STRIDE)
        
        if len(crops) == 0:
            raise ValueError("No player crops collected for team classification training")
        
        logging.info(f"Collected {len(crops)} player crops for training")
        
        # Initialize and train team classifier
        team_classifier = TeamClassifier(device=DEVICE)
        team_classifier.fit(crops)
        
        logging.info("Team classifier training completed")
        return team_classifier
        
    except ImportError:
        logging.error("Could not import sports library. Please install it from GitHub.")
        raise
    except Exception as e:
        logging.error(f"Error training team classifier: {e}")
        raise

def process_video_with_analysis(video_path: str, output_path: str, 
                               detection_model, team_classifier,
                               field_model=None, sam_model=None):
    """
    Process video with complete analysis including possession and movement tracking
    
    Args:
        video_path: Input video path
        output_path: Output video path
        detection_model: Player detection model
        team_classifier: Trained team classifier
        field_model: Field detection model (optional)
        sam_model: SAM model (optional)
    """
    logging.info(f"Processing video: {video_path}")
    
    # Initialize video processor
    processor = VideoProcessor(detection_model, field_model, sam_model)
    
    # Initialize analyzers
    possession_analyzer = BallPossessionAnalyzer()
    movement_analyzer = PlayerMovementAnalyzer()
    formation_analyzer = FormationAnalyzer()
    
    # Process video with analysis
    import supervision as sv
    import cv2
    from tqdm import tqdm
    
    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path)
    
    # Setup video writer
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_info.fps,
        (video_info.width, video_info.height)
    )
    
    try:
        frame_number = 0
        for frame in tqdm(frame_generator, total=video_info.total_frames, 
                         desc="Processing video with analysis"):
            
            # Detect objects
            detections = processor.detect_objects(frame)
            detections = processor.filter_detections(detections)
            
            # Separate detections by type
            detections_dict = processor.separate_detections(detections)
            
            # Update tracking for players and goalkeepers
            tracked_objects = sv.Detections.merge([
                detections_dict['players'], 
                detections_dict['goalkeepers']
            ])
            tracked_objects = processor.update_tracking(tracked_objects)
            
            # Update detections with tracked objects
            detections_dict['players'] = tracked_objects[tracked_objects.class_id == PLAYER_ID]
            detections_dict['goalkeepers'] = tracked_objects[tracked_objects.class_id == GOALKEEPER_ID]
            
            # Assign team IDs
            player_team_ids, goalkeeper_team_ids = processor.assign_team_ids(
                frame, tracked_objects, team_classifier, frame_number
            )
            
            team_assignments = {
                'players': player_team_ids,
                'goalkeepers': goalkeeper_team_ids
            }
            
            # Update analysis
            if len(detections_dict['players']) > 0 and len(player_team_ids) > 0:
                # Update possession analysis
                possession_analyzer.update_possession(
                    detections_dict['ball'], 
                    detections_dict['players'], 
                    player_team_ids,
                    frame_number
                )
                
                # Update movement analysis
                movement_analyzer.update_positions(
                    detections_dict['players'],
                    player_team_ids,
                    frame_number
                )
                
                # Update formation analysis
                formation_analyzer.analyze_formation(
                    detections_dict['players'],
                    player_team_ids,
                    frame_number
                )
            
            # Annotate frame
            annotated_frame = processor.annotate_frame(frame, detections_dict, team_assignments)
            
            # Add possession information to frame
            possession_percentages = possession_analyzer.get_possession_percentages(frame_number + 1)
            
            # Draw possession info on frame
            y_offset = 30
            for team_id, percentage in possession_percentages.items():
                color = COLORS['team_0_players'] if team_id == 0 else COLORS['team_1_players']
                text = f"Team {team_id}: {percentage:.1f}%"
                
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           tuple(int(color[i:i+2], 16) for i in (1, 3, 5))[::-1], 2)
                y_offset += 30
            
            # Write frame
            video_writer.write(annotated_frame)
            frame_number += 1
    
    finally:
        video_writer.release()
        
        # Generate final statistics
        from analysis import calculate_team_statistics
        final_stats = calculate_team_statistics(
            possession_analyzer, movement_analyzer, frame_number
        )
        
        # Log final statistics
        logging.info("Final Analysis Results:")
        logging.info(f"Team 0 possession: {final_stats['possession'][0]:.1f}%")
        logging.info(f"Team 1 possession: {final_stats['possession'][1]:.1f}%")
        logging.info(f"Total possession switches: {final_stats['possession_switches']}")
        
        logging.info(f"Video processing completed. Output saved to: {output_path}")
        
        return final_stats

def main():
    """Main function to run the complete football analysis workflow"""
    
    # Setup
    setup_logging()
    setup_environment()
    
    logging.info("Starting Football Analysis Pipeline")
    
    try:
        # Load models
        detection_model, field_model, sam_model = load_models()
        
        # Train team classifier
        team_classifier = train_team_classifier(detection_model)
        
        # Process video with complete analysis
        final_stats = process_video_with_analysis(
            video_path=SOURCE_VIDEO_PATH,
            output_path=OUTPUT_VIDEO_PATH,
            detection_model=detection_model,
            team_classifier=team_classifier,
            field_model=field_model,
            sam_model=sam_model
        )
        
        logging.info("Football analysis pipeline completed successfully")
        return final_stats
        
    except Exception as e:
        logging.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    stats = main() 