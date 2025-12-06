"""
🎬 Replay System - Record and playback training episodes
"""

import json
import gzip
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np


class ReplayRecorder:
    """Records episodes for later playback."""
    
    def __init__(self, save_dir="replays"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.current_recording = None
        self.is_recording = False
    
    def start_recording(self, metadata=None):
        """Start recording a new replay."""
        self.current_recording = {
            'metadata': metadata or {},
            'frames': [],
            'start_time': datetime.now().isoformat(),
        }
        self.is_recording = True
    
    def record_frame(self, state, action, reward, info=None):
        """Record a single frame."""
        if not self.is_recording:
            return
        
        frame = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': int(action) if isinstance(action, (np.integer, np.ndarray)) else action,
            'reward': float(reward),
            'info': info or {}
        }
        self.current_recording['frames'].append(frame)
    
    def stop_recording(self, filename=None):
        """Stop recording and save replay."""
        if not self.is_recording:
            return None
        
        self.current_recording['end_time'] = datetime.now().isoformat()
        self.current_recording['duration'] = len(self.current_recording['frames'])
        
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"replay_{timestamp}.replay"
        
        filepath = self.save_dir / filename
        
        # Compress and save
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(self.current_recording, f)
        
        self.is_recording = False
        self.current_recording = None
        
        return filepath
    
    def list_replays(self):
        """List all saved replays."""
        replays = []
        for replay_file in self.save_dir.glob("*.replay"):
            with gzip.open(replay_file, 'rb') as f:
                data = pickle.load(f)
                replays.append({
                    'filename': replay_file.name,
                    'path': str(replay_file),
                    'metadata': data.get('metadata', {}),
                    'duration': data.get('duration', 0),
                    'start_time': data.get('start_time', 'Unknown')
                })
        return replays


class ReplayPlayer:
    """Plays back recorded episodes."""
    
    def __init__(self):
        self.replay_data = None
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
    
    def load_replay(self, filepath):
        """Load a replay file."""
        with gzip.open(filepath, 'rb') as f:
            self.replay_data = pickle.load(f)
        self.current_frame = 0
        return self.replay_data
    
    def get_frame(self, frame_idx=None):
        """Get a specific frame."""
        if self.replay_data is None:
            return None
        
        if frame_idx is None:
            frame_idx = self.current_frame
        
        if 0 <= frame_idx < len(self.replay_data['frames']):
            return self.replay_data['frames'][frame_idx]
        return None
    
    def next_frame(self):
        """Get next frame."""
        if self.replay_data is None:
            return None
        
        if self.current_frame < len(self.replay_data['frames']) - 1:
            self.current_frame += 1
            return self.get_frame()
        return None
    
    def previous_frame(self):
        """Get previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            return self.get_frame()
        return None
    
    def seek(self, frame_idx):
        """Seek to specific frame."""
        if 0 <= frame_idx < len(self.replay_data['frames']):
            self.current_frame = frame_idx
            return self.get_frame()
        return None
    
    def reset(self):
        """Reset to beginning."""
        self.current_frame = 0
    
    def get_metadata(self):
        """Get replay metadata."""
        if self.replay_data is None:
            return {}
        return self.replay_data.get('metadata', {})
    
    def get_statistics(self):
        """Calculate replay statistics."""
        if self.replay_data is None:
            return {}
        
        frames = self.replay_data['frames']
        rewards = [f['reward'] for f in frames]
        actions = [f['action'] for f in frames]
        
        return {
            'total_frames': len(frames),
            'total_reward': sum(rewards),
            'average_reward': np.mean(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'action_distribution': {
                action: actions.count(action) / len(actions)
                for action in set(actions)
            }
        }


# Example usage
if __name__ == "__main__":
    # Recording
    recorder = ReplayRecorder()
    recorder.start_recording(metadata={'agent': 'DQN', 'episode': 1})
    
    for i in range(100):
        state = np.random.rand(10)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        recorder.record_frame(state, action, reward)
    
    filepath = recorder.stop_recording()
    print(f"Saved replay to: {filepath}")
    
    # Playback
    player = ReplayPlayer()
    player.load_replay(filepath)
    
    print("Metadata:", player.get_metadata())
    print("Statistics:", player.get_statistics())
    
    # Play first 10 frames
    for i in range(10):
        frame = player.next_frame()
        if frame:
            print(f"Frame {i}: Action={frame['action']}, Reward={frame['reward']:.2f}")
