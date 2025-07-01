"""
Analysis Module for Football Video Processing
Handles ball possession, team statistics, and advanced analytics
"""

import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Optional
import logging

from config import *

class BallPossessionAnalyzer:
    """
    Class for analyzing ball possession and control
    """
    
    def __init__(self):
        self.possession_history = []  # List of (frame_number, team_id) tuples
        self.team_possession_time = {0: 0, 1: 0}  # Total frames each team has possession
        self.current_possession = None  # Current team in possession
        
    def update_possession(self, ball_detections: sv.Detections, 
                         player_detections: sv.Detections, 
                         player_team_ids: List[int],
                         frame_number: int) -> Optional[int]:
        """
        Update ball possession based on ball and player positions
        
        Args:
            ball_detections: Ball detection results
            player_detections: Player detection results
            player_team_ids: Team assignments for players
            frame_number: Current frame number
            
        Returns:
            Team ID in possession (0, 1) or None if no possession
        """
        if len(ball_detections) == 0 or len(player_detections) == 0:
            return None
        
        # Get ball position (center of bounding box)
        ball_center = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0]
        
        # Get player positions
        player_positions = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        # Calculate distances from ball to all players
        distances = np.linalg.norm(player_positions - ball_center, axis=1)
        
        # Find closest player
        closest_player_idx = np.argmin(distances)
        closest_distance = distances[closest_player_idx]
        
        # Check if closest player is within possession threshold
        if closest_distance <= BALL_CONTROL_DISTANCE_THRESHOLD:
            possessing_team = player_team_ids[closest_player_idx]
            
            # Update possession statistics
            self.current_possession = possessing_team
            self.team_possession_time[possessing_team] += 1
            self.possession_history.append((frame_number, possessing_team))
            
            return possessing_team
        
        return None
    
    def get_possession_percentages(self, total_frames: int) -> Dict[int, float]:
        """
        Calculate possession percentages for each team
        
        Args:
            total_frames: Total number of frames processed
            
        Returns:
            Dictionary with possession percentages
        """
        total_possession_frames = sum(self.team_possession_time.values())
        
        if total_possession_frames == 0:
            return {0: 0.0, 1: 0.0}
        
        return {
            0: (self.team_possession_time[0] / total_possession_frames) * 100,
            1: (self.team_possession_time[1] / total_possession_frames) * 100
        }
    
    def get_possession_switches(self) -> List[Tuple[int, int, int]]:
        """
        Get list of possession switches with frame numbers
        
        Returns:
            List of (frame_number, from_team, to_team) tuples
        """
        switches = []
        
        if len(self.possession_history) < 2:
            return switches
        
        current_team = self.possession_history[0][1]
        
        for frame_number, team_id in self.possession_history[1:]:
            if team_id != current_team:
                switches.append((frame_number, current_team, team_id))
                current_team = team_id
        
        return switches

class PlayerMovementAnalyzer:
    """
    Class for analyzing player movement patterns and statistics
    """
    
    def __init__(self):
        self.player_positions = {}  # tracker_id -> [(frame, x, y), ...]
        self.player_distances = {}  # tracker_id -> total_distance
        self.team_coverage = {0: [], 1: []}  # team_id -> [positions]
    
    def update_positions(self, player_detections: sv.Detections, 
                        player_team_ids: List[int], 
                        frame_number: int):
        """
        Update player position tracking
        
        Args:
            player_detections: Player detection results
            player_team_ids: Team assignments for players
            frame_number: Current frame number
        """
        if len(player_detections) == 0:
            return
        
        positions = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        for i, (tracker_id, position, team_id) in enumerate(
            zip(player_detections.tracker_id, positions, player_team_ids)
        ):
            # Update player position history
            if tracker_id not in self.player_positions:
                self.player_positions[tracker_id] = []
                self.player_distances[tracker_id] = 0.0
            
            self.player_positions[tracker_id].append((frame_number, position[0], position[1]))
            
            # Calculate distance traveled since last frame
            if len(self.player_positions[tracker_id]) > 1:
                prev_pos = np.array(self.player_positions[tracker_id][-2][1:])
                curr_pos = np.array(self.player_positions[tracker_id][-1][1:])
                distance = np.linalg.norm(curr_pos - prev_pos)
                self.player_distances[tracker_id] += distance
            
            # Update team coverage
            self.team_coverage[team_id].append(position)
    
    def get_player_statistics(self) -> Dict[int, Dict[str, float]]:
        """
        Get movement statistics for each player
        
        Returns:
            Dictionary mapping tracker_id to statistics
        """
        stats = {}
        
        for tracker_id, positions in self.player_positions.items():
            if len(positions) < 2:
                continue
            
            # Extract position coordinates
            coords = np.array([(pos[1], pos[2]) for pos in positions])
            
            # Calculate statistics
            total_distance = self.player_distances.get(tracker_id, 0.0)
            avg_speed = total_distance / len(positions) if len(positions) > 0 else 0.0
            
            # Calculate area covered (convex hull area)
            if len(coords) >= 3:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(coords)
                    area_covered = hull.volume  # In 2D, volume is area
                except:
                    area_covered = 0.0
            else:
                area_covered = 0.0
            
            stats[tracker_id] = {
                'total_distance': total_distance,
                'average_speed': avg_speed,
                'area_covered': area_covered,
                'total_frames': len(positions)
            }
        
        return stats
    
    def get_team_heatmap_data(self, team_id: int) -> np.ndarray:
        """
        Get position data for creating team heatmaps
        
        Args:
            team_id: Team ID (0 or 1)
            
        Returns:
            Array of positions for the team
        """
        positions = self.team_coverage.get(team_id, [])
        return np.array(positions) if positions else np.array([]).reshape(0, 2)

class FormationAnalyzer:
    """
    Class for analyzing team formations and tactical patterns
    """
    
    def __init__(self):
        self.formation_history = {0: [], 1: []}  # team_id -> [formation_data]
    
    def analyze_formation(self, player_detections: sv.Detections, 
                         player_team_ids: List[int], 
                         frame_number: int) -> Dict[int, Dict]:
        """
        Analyze current team formations
        
        Args:
            player_detections: Player detection results
            player_team_ids: Team assignments for players
            frame_number: Current frame number
            
        Returns:
            Dictionary with formation analysis for each team
        """
        formations = {}
        
        for team_id in [0, 1]:
            team_mask = np.array(player_team_ids) == team_id
            if not np.any(team_mask):
                formations[team_id] = {'players': 0, 'centroid': None, 'spread': 0.0}
                continue
            
            team_players = player_detections[team_mask]
            positions = team_players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            
            # Calculate team centroid
            centroid = np.mean(positions, axis=0)
            
            # Calculate team spread (average distance from centroid)
            distances = np.linalg.norm(positions - centroid, axis=1)
            spread = np.mean(distances)
            
            formations[team_id] = {
                'players': len(positions),
                'centroid': centroid,
                'spread': spread,
                'positions': positions.tolist()
            }
            
            # Store in history
            self.formation_history[team_id].append({
                'frame': frame_number,
                'centroid': centroid.tolist(),
                'spread': spread,
                'player_count': len(positions)
            })
        
        return formations
    
    def get_formation_timeline(self, team_id: int) -> List[Dict]:
        """
        Get formation changes over time for a team
        
        Args:
            team_id: Team ID (0 or 1)
            
        Returns:
            List of formation data over time
        """
        return self.formation_history.get(team_id, [])

class GameEventDetector:
    """
    Class for detecting significant game events
    """
    
    def __init__(self):
        self.events = []  # List of detected events
        self.last_ball_position = None
        self.possession_analyzer = BallPossessionAnalyzer()
    
    def detect_events(self, ball_detections: sv.Detections, 
                     player_detections: sv.Detections,
                     player_team_ids: List[int],
                     frame_number: int) -> List[Dict]:
        """
        Detect game events in current frame
        
        Args:
            ball_detections: Ball detection results
            player_detections: Player detection results
            player_team_ids: Team assignments for players
            frame_number: Current frame number
            
        Returns:
            List of events detected in this frame
        """
        events = []
        
        if len(ball_detections) == 0:
            return events
        
        current_ball_pos = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0]
        
        # Detect rapid ball movement (potential pass/shot)
        if self.last_ball_position is not None:
            ball_movement = np.linalg.norm(current_ball_pos - self.last_ball_position)
            
            if ball_movement > 50:  # Threshold for significant movement
                events.append({
                    'type': 'rapid_ball_movement',
                    'frame': frame_number,
                    'movement_distance': float(ball_movement),
                    'position': current_ball_pos.tolist()
                })
        
        # Detect possession changes
        current_possession = self.possession_analyzer.update_possession(
            ball_detections, player_detections, player_team_ids, frame_number
        )
        
        # Store current position for next frame
        self.last_ball_position = current_ball_pos
        
        return events
    
    def get_all_events(self) -> List[Dict]:
        """Get all detected events"""
        return self.events

def calculate_team_statistics(possession_analyzer: BallPossessionAnalyzer,
                            movement_analyzer: PlayerMovementAnalyzer,
                            total_frames: int) -> Dict:
    """
    Calculate comprehensive team statistics
    
    Args:
        possession_analyzer: Ball possession analyzer
        movement_analyzer: Player movement analyzer
        total_frames: Total frames processed
        
    Returns:
        Dictionary with comprehensive statistics
    """
    stats = {
        'possession': possession_analyzer.get_possession_percentages(total_frames),
        'possession_switches': len(possession_analyzer.get_possession_switches()),
        'player_stats': movement_analyzer.get_player_statistics(),
        'total_frames': total_frames
    }
    
    # Calculate team-level movement statistics
    for team_id in [0, 1]:
        team_players = [pid for pid, pstats in stats['player_stats'].items() 
                       if pid in movement_analyzer.player_positions]
        
        if team_players:
            team_distances = [stats['player_stats'][pid]['total_distance'] 
                            for pid in team_players if pid in stats['player_stats']]
            stats[f'team_{team_id}_avg_distance'] = np.mean(team_distances) if team_distances else 0.0
            stats[f'team_{team_id}_total_distance'] = np.sum(team_distances) if team_distances else 0.0
    
    return stats 