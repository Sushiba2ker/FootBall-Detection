"""
Video Processing Module for Football Analysis
Handles video loading, object detection, tracking, and basic processing workflows
"""

import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Optional, Generator
from tqdm import tqdm
import logging

from config import *

class VideoProcessor:
    """
    Main class for processing football videos with object detection and tracking
    """
    
    def __init__(self, detection_model, field_model=None, sam_model=None):
        """
        Initialize video processor with models
        
        Args:
            detection_model: Player/ball detection model (from Roboflow)
            field_model: Field keypoint detection model (optional)
            sam_model: SAM model for segmentation (optional)
        """
        self.detection_model = detection_model
        self.field_model = field_model
        self.sam_model = sam_model
        
        # Initialize tracker
        self.tracker = sv.ByteTrack()
        self.tracker.reset()
        
        # Initialize annotators
        self._setup_annotators()
        
        # Team classification cache
        self.team_cache = {}  # tracker_id -> team_id
        self.last_classification_frame = {}  # tracker_id -> frame_number
        
        logging.info("VideoProcessor initialized successfully")
    
    def _setup_annotators(self):
        """Setup all annotators for different object types and teams"""
        
        # Players Team 0
        self.ellipse_annotator_p0 = sv.EllipseAnnotator(
            color=sv.Color.from_hex(COLORS['team_0_players']),
            thickness=ELLIPSE_THICKNESS
        )
        self.label_annotator_p0 = sv.LabelAnnotator(
            color=sv.Color.from_hex(COLORS['team_0_players']),
            text_color=sv.Color.from_hex(COLORS['text_dark']),
            text_position=sv.Position.BOTTOM_CENTER
        )
        
        # Players Team 1
        self.ellipse_annotator_p1 = sv.EllipseAnnotator(
            color=sv.Color.from_hex(COLORS['team_1_players']),
            thickness=ELLIPSE_THICKNESS
        )
        self.label_annotator_p1 = sv.LabelAnnotator(
            color=sv.Color.from_hex(COLORS['team_1_players']),
            text_color=sv.Color.from_hex(COLORS['text_dark']),
            text_position=sv.Position.BOTTOM_CENTER
        )
        
        # Goalkeepers Team 0
        self.ellipse_annotator_g0 = sv.EllipseAnnotator(
            color=sv.Color.from_hex(COLORS['team_0_goalkeeper']),
            thickness=ELLIPSE_THICKNESS
        )
        self.label_annotator_g0 = sv.LabelAnnotator(
            color=sv.Color.from_hex(COLORS['team_0_goalkeeper']),
            text_color=sv.Color.from_hex(COLORS['text_dark']),
            text_position=sv.Position.BOTTOM_CENTER
        )
        
        # Goalkeepers Team 1
        self.ellipse_annotator_g1 = sv.EllipseAnnotator(
            color=sv.Color.from_hex(COLORS['team_1_goalkeeper']),
            thickness=ELLIPSE_THICKNESS
        )
        self.label_annotator_g1 = sv.LabelAnnotator(
            color=sv.Color.from_hex(COLORS['team_1_goalkeeper']),
            text_color=sv.Color.from_hex(COLORS['text_dark']),
            text_position=sv.Position.BOTTOM_CENTER
        )
        
        # Referees
        self.ellipse_annotator_ref = sv.EllipseAnnotator(
            color=sv.Color.from_hex(COLORS['referee']),
            thickness=ELLIPSE_THICKNESS
        )
        self.label_annotator_ref = sv.LabelAnnotator(
            color=sv.Color.from_hex(COLORS['referee']),
            text_color=sv.Color.from_hex(COLORS['text_light']),
            text_position=sv.Position.BOTTOM_CENTER
        )
        
        # Ball
        self.triangle_annotator_ball = sv.TriangleAnnotator(
            color=sv.Color.from_hex(COLORS['ball']),
            base=TRIANGLE_BASE,
            height=TRIANGLE_HEIGHT,
            outline_thickness=TRIANGLE_OUTLINE_THICKNESS
        )
    
    def detect_objects(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect objects in a single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            supervision.Detections object with all detected objects
        """
        result = self.detection_model.infer(frame, confidence=CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_inference(result)
        return detections
    
    def filter_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Filter detections to remove noise based on size and other criteria
        
        Args:
            detections: Input detections
            
        Returns:
            Filtered detections
        """
        # Calculate areas
        areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        
        # Filter by minimum size (except for ball)
        valid_mask = (areas >= MIN_PLAYER_SIZE) | (detections.class_id == BALL_ID)
        
        return detections[valid_mask]
    
    def separate_detections(self, detections: sv.Detections) -> Dict[str, sv.Detections]:
        """
        Separate detections by object type
        
        Args:
            detections: All detections
            
        Returns:
            Dictionary with separated detections by type
        """
        return {
            'ball': detections[detections.class_id == BALL_ID],
            'players': detections[detections.class_id == PLAYER_ID],
            'goalkeepers': detections[detections.class_id == GOALKEEPER_ID],
            'referees': detections[detections.class_id == REFEREE_ID]
        }
    
    def update_tracking(self, detections: sv.Detections) -> sv.Detections:
        """
        Update object tracking for players and goalkeepers
        
        Args:
            detections: Detections to track (should only include players and goalkeepers)
            
        Returns:
            Detections with updated tracker IDs
        """
        # Apply NMS first
        detections = detections.with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections=detections)
        
        return detections
    
    def resolve_goalkeepers_team_id(self, players: sv.Detections, goalkeepers: sv.Detections, 
                                   frame_width: int) -> np.ndarray:
        """
        Determine team ID for goalkeepers based on player positions
        
        Args:
            players: Player detections with team assignments
            goalkeepers: Goalkeeper detections
            frame_width: Width of the video frame
            
        Returns:
            Array of team IDs for goalkeepers
        """
        if len(players) == 0:
            return np.zeros(len(goalkeepers), dtype=int)
        
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        team_ids = players.class_id
        
        # Separate players by field half
        left_half_mask = players_xy[:, 0] < frame_width / 2
        right_half_mask = ~left_half_mask
        
        # Determine dominant team in each half
        left_team_id = np.bincount(team_ids[left_half_mask]).argmax() if any(left_half_mask) else 0
        right_team_id = np.bincount(team_ids[right_half_mask]).argmax() if any(right_half_mask) else 1
        
        # Assign goalkeeper teams based on their position
        goalkeepers_team_id = []
        for gk_xy in goalkeepers_xy:
            is_left_side = gk_xy[0] < frame_width / 2
            goalkeepers_team_id.append(left_team_id if is_left_side else right_team_id)
        
        return np.array(goalkeepers_team_id)
    
    def assign_team_ids(self, frame: np.ndarray, tracked_detections: sv.Detections, 
                       team_classifier, frame_number: int) -> Tuple[List[int], List[int]]:
        """
        Assign team IDs to players and goalkeepers with caching
        
        Args:
            frame: Current frame
            tracked_detections: Tracked detections containing players and goalkeepers
            team_classifier: Team classification model
            frame_number: Current frame number
            
        Returns:
            Tuple of (player_team_ids, goalkeeper_team_ids)
        """
        # Separate players and goalkeepers
        players = tracked_detections[tracked_detections.class_id == PLAYER_ID]
        goalkeepers = tracked_detections[tracked_detections.class_id == GOALKEEPER_ID]
        
        # Assign team IDs to players
        player_team_ids = []
        for i, tracker_id in enumerate(players.tracker_id):
            # Check if we need to reclassify
            should_classify = (
                tracker_id not in self.team_cache or
                tracker_id not in self.last_classification_frame or
                frame_number - self.last_classification_frame[tracker_id] >= TEAM_CLASSIFICATION_INTERVAL
            )
            
            if should_classify:
                # Classify team for this player
                crop = sv.crop_image(frame, players.xyxy[i])
                team_id = team_classifier.predict([crop])[0]
                
                # Update cache
                self.team_cache[tracker_id] = team_id
                self.last_classification_frame[tracker_id] = frame_number
            
            player_team_ids.append(self.team_cache[tracker_id])
        
        # Assign team IDs to goalkeepers
        if len(goalkeepers) > 0:
            # Update player class_ids with team assignments for goalkeeper resolution
            players_with_teams = players.copy()
            if len(player_team_ids) > 0:
                players_with_teams.class_id = np.array(player_team_ids)
            
            goalkeeper_team_ids = self.resolve_goalkeepers_team_id(
                players_with_teams, goalkeepers, frame.shape[1]
            ).tolist()
        else:
            goalkeeper_team_ids = []
        
        return player_team_ids, goalkeeper_team_ids
    
    def annotate_frame(self, frame: np.ndarray, detections_dict: Dict[str, sv.Detections],
                      team_assignments: Dict[str, List[int]]) -> np.ndarray:
        """
        Annotate frame with all detections and team colors
        
        Args:
            frame: Input frame
            detections_dict: Dictionary containing separated detections
            team_assignments: Dictionary containing team assignments
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Annotate ball
        if len(detections_dict['ball']) > 0:
            # Add padding to ball detection
            ball_detections = detections_dict['ball'].copy()
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            annotated_frame = self.triangle_annotator_ball.annotate(
                scene=annotated_frame, detections=ball_detections
            )
        
        # Annotate players by team
        players = detections_dict['players']
        player_team_ids = team_assignments.get('players', [])
        
        if len(players) > 0 and len(player_team_ids) > 0:
            # Team 0 players
            team_0_mask = np.array(player_team_ids) == 0
            if np.any(team_0_mask):
                team_0_players = players[team_0_mask]
                labels = [f"#{tracker_id}" for tracker_id in team_0_players.tracker_id]
                annotated_frame = self.ellipse_annotator_p0.annotate(
                    scene=annotated_frame, detections=team_0_players
                )
                annotated_frame = self.label_annotator_p0.annotate(
                    scene=annotated_frame, detections=team_0_players, labels=labels
                )
            
            # Team 1 players
            team_1_mask = np.array(player_team_ids) == 1
            if np.any(team_1_mask):
                team_1_players = players[team_1_mask]
                labels = [f"#{tracker_id}" for tracker_id in team_1_players.tracker_id]
                annotated_frame = self.ellipse_annotator_p1.annotate(
                    scene=annotated_frame, detections=team_1_players
                )
                annotated_frame = self.label_annotator_p1.annotate(
                    scene=annotated_frame, detections=team_1_players, labels=labels
                )
        
        # Annotate goalkeepers by team
        goalkeepers = detections_dict['goalkeepers']
        goalkeeper_team_ids = team_assignments.get('goalkeepers', [])
        
        if len(goalkeepers) > 0 and len(goalkeeper_team_ids) > 0:
            # Team 0 goalkeepers
            team_0_mask = np.array(goalkeeper_team_ids) == 0
            if np.any(team_0_mask):
                team_0_goalkeepers = goalkeepers[team_0_mask]
                labels = [f"#{tracker_id}" for tracker_id in team_0_goalkeepers.tracker_id]
                annotated_frame = self.ellipse_annotator_g0.annotate(
                    scene=annotated_frame, detections=team_0_goalkeepers
                )
                annotated_frame = self.label_annotator_g0.annotate(
                    scene=annotated_frame, detections=team_0_goalkeepers, labels=labels
                )
            
            # Team 1 goalkeepers
            team_1_mask = np.array(goalkeeper_team_ids) == 1
            if np.any(team_1_mask):
                team_1_goalkeepers = goalkeepers[team_1_mask]
                labels = [f"#{tracker_id}" for tracker_id in team_1_goalkeepers.tracker_id]
                annotated_frame = self.ellipse_annotator_g1.annotate(
                    scene=annotated_frame, detections=team_1_goalkeepers
                )
                annotated_frame = self.label_annotator_g1.annotate(
                    scene=annotated_frame, detections=team_1_goalkeepers, labels=labels
                )
        
        # Annotate referees
        if len(detections_dict['referees']) > 0:
            labels = [f"#{i+1}" for i in range(len(detections_dict['referees']))]
            annotated_frame = self.ellipse_annotator_ref.annotate(
                scene=annotated_frame, detections=detections_dict['referees']
            )
            annotated_frame = self.label_annotator_ref.annotate(
                scene=annotated_frame, detections=detections_dict['referees'], labels=labels
            )
        
        return annotated_frame
    
    def process_video(self, video_path: str, output_path: str, team_classifier,
                     progress_callback=None) -> None:
        """
        Process entire video with detection, tracking, and annotation
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            team_classifier: Team classification model
            progress_callback: Optional callback for progress updates
        """
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
                            desc="Processing video"):
                
                # Detect objects
                detections = self.detect_objects(frame)
                detections = self.filter_detections(detections)
                
                # Separate detections by type
                detections_dict = self.separate_detections(detections)
                
                # Update tracking for players and goalkeepers
                tracked_objects = sv.Detections.merge([
                    detections_dict['players'], 
                    detections_dict['goalkeepers']
                ])
                tracked_objects = self.update_tracking(tracked_objects)
                
                # Update detections with tracked objects
                detections_dict['players'] = tracked_objects[tracked_objects.class_id == PLAYER_ID]
                detections_dict['goalkeepers'] = tracked_objects[tracked_objects.class_id == GOALKEEPER_ID]
                
                # Assign team IDs
                player_team_ids, goalkeeper_team_ids = self.assign_team_ids(
                    frame, tracked_objects, team_classifier, frame_number
                )
                
                team_assignments = {
                    'players': player_team_ids,
                    'goalkeepers': goalkeeper_team_ids
                }
                
                # Annotate frame
                annotated_frame = self.annotate_frame(frame, detections_dict, team_assignments)
                
                # Write frame
                video_writer.write(annotated_frame)
                
                frame_number += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(frame_number, video_info.total_frames)
        
        finally:
            video_writer.release()
            logging.info(f"Video processing completed. Output saved to: {output_path}")

def collect_team_training_crops(video_path: str, detection_model, stride: int = STRIDE) -> List[np.ndarray]:
    """
    Collect player crops from video for team classifier training
    
    Args:
        video_path: Path to video file
        detection_model: Object detection model
        stride: Frame stride for sampling
        
    Returns:
        List of player crop images
    """
    frame_generator = sv.get_video_frames_generator(video_path, stride=stride)
    crops = []
    
    for frame in tqdm(frame_generator, desc='Collecting team training crops'):
        result = detection_model.infer(frame, confidence=CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_inference(result)
        
        # Get player detections
        players_detections = detections[detections.class_id == PLAYER_ID]
        
        # Crop player images
        player_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        crops.extend(player_crops)
    
    logging.info(f"Collected {len(crops)} player crops for team classification training")
    return crops 