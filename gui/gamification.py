"""
🏆 Gamification System - Achievements, badges, and challenges
"""

import json
from pathlib import Path
from datetime import datetime
from enum import Enum


class AchievementType(Enum):
    """Types of achievements."""
    TRAINING = "training"
    PERFORMANCE = "performance"
    MILESTONE = "milestone"
    SPECIAL = "special"


class Achievement:
    """Individual achievement."""
    
    def __init__(self, id, name, description, icon, requirement, points):
        self.id = id
        self.name = name
        self.description = description
        self.icon = icon
        self.requirement = requirement
        self.points = points
        self.unlocked = False
        self.unlock_date = None
    
    def check_unlock(self, stats):
        """Check if achievement should be unlocked."""
        if self.unlocked:
            return False
        
        # Check requirement
        if self.requirement(stats):
            self.unlock()
            return True
        return False
    
    def unlock(self):
        """Unlock this achievement."""
        self.unlocked = True
        self.unlock_date = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'points': self.points,
            'unlocked': self.unlocked,
            'unlock_date': self.unlock_date
        }


class GamificationSystem:
    """Manages achievements, XP, and challenges."""
    
    def __init__(self, save_dir="gamification"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.achievements = []
        self.total_xp = 0
        self.level = 1
        self.challenges_completed = []
        
        self._initialize_achievements()
        self._load_progress()
    
    def _initialize_achievements(self):
        """Initialize all achievements."""
        self.achievements = [
            # Training milestones
            Achievement(
                "first_training",
                "🎓 First Training",
                "Complete your first training episode",
                "🎓",
                lambda s: s.get('total_episodes', 0) >= 1,
                10
            ),
            Achievement(
                "100_episodes",
                "💯 Century",
                "Complete 100 training episodes",
                "💯",
                lambda s: s.get('total_episodes', 0) >= 100,
                50
            ),
            Achievement(
                "1000_episodes",
                "🏆 Millennium",
                "Complete 1000 training episodes",
                "🏆",
                lambda s: s.get('total_episodes', 0) >= 1000,
                200
            ),
            
            # Performance achievements
            Achievement(
                "first_success",
                "✅ First Success",
                "Successfully reach the goal",
                "✅",
                lambda s: s.get('successes', 0) >= 1,
                20
            ),
            Achievement(
                "90_percent_success",
                "🌟 Master Navigator",
                "Achieve 90% success rate over 100 episodes",
                "🌟",
                lambda s: s.get('success_rate', 0) >= 0.9,
                100
            ),
            Achievement(
                "perfect_episode",
                "💎 Perfect Run",
                "Complete an episode with maximum efficiency",
                "💎",
                lambda s: s.get('max_reward', float('-inf')) >= 1000,
                150
            ),
            
            # Speed achievements
            Achievement(
                "speed_demon",
                "⚡ Speed Demon",
                "Complete episode in under 50 steps",
                "⚡",
                lambda s: s.get('min_steps', float('inf')) <= 50,
                75
            ),
            
            # Special achievements
            Achievement(
                "night_owl",
                "🦉 Night Owl",
                "Train after midnight",
                "🦉",
                lambda s: s.get('trained_at_night', False),
                25
            ),
            Achievement(
                "marathon",
                "🏃 Marathon Runner",
                "Train for 24 hours continuously",
                "🏃",
                lambda s: s.get('training_hours', 0) >= 24,
                300
            ),
        ]
    
    def check_achievements(self, stats):
        """Check all achievements against current stats."""
        newly_unlocked = []
        
        for achievement in self.achievements:
            if achievement.check_unlock(stats):
                self.total_xp += achievement.points
                newly_unlocked.append(achievement)
                print(f"🎉 Achievement Unlocked: {achievement.icon} {achievement.name}")
                print(f"   +{achievement.points} XP")
        
        # Check for level up
        if newly_unlocked:
            self._check_level_up()
            self._save_progress()
        
        return newly_unlocked
    
    def _check_level_up(self):
        """Check if player leveled up."""
        # XP required = level^2 * 100
        xp_required = self.level ** 2 * 100
        
        if self.total_xp >= xp_required:
            self.level += 1
            print(f"🎊 LEVEL UP! You are now level {self.level}!")
            return True
        return False
    
    def get_progress(self):
        """Get current progress."""
        unlocked = sum(1 for a in self.achievements if a.unlocked)
        total = len(self.achievements)
        
        return {
            'level': self.level,
            'xp': self.total_xp,
            'xp_to_next_level': self.level ** 2 * 100 - self.total_xp,
            'achievements_unlocked': unlocked,
            'achievements_total': total,
            'completion_percentage': (unlocked / total) * 100 if total > 0 else 0
        }
    
    def get_leaderboard(self):
        """Get leaderboard data."""
        # TODO: Implement global/local leaderboard
        return {
            'local': [
                {'name': 'You', 'level': self.level, 'xp': self.total_xp},
            ],
            'global': []
        }
    
    def _save_progress(self):
        """Save progress to file."""
        data = {
            'xp': self.total_xp,
            'level': self.level,
            'achievements': [a.to_dict() for a in self.achievements],
            'challenges_completed': self.challenges_completed,
            'last_updated': datetime.now().isoformat()
        }
        
        save_file = self.save_dir / "progress.json"
        with open(save_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_progress(self):
        """Load progress from file."""
        save_file = self.save_dir / "progress.json"
        
        if not save_file.exists():
            return
        
        try:
            with open(save_file, 'r') as f:
                data = json.load(f)
            
            self.total_xp = data.get('xp', 0)
            self.level = data.get('level', 1)
            self.challenges_completed = data.get('challenges_completed', [])
            
            # Restore achievement unlock status
            saved_achievements = {a['id']: a for a in data.get('achievements', [])}
            for achievement in self.achievements:
                if achievement.id in saved_achievements:
                    saved = saved_achievements[achievement.id]
                    achievement.unlocked = saved['unlocked']
                    achievement.unlock_date = saved['unlock_date']
        
        except Exception as e:
            print(f"Error loading progress: {e}")
    
    def get_daily_challenge(self):
        """Get today's daily challenge."""
        # TODO: Generate daily challenge based on date
        return {
            'title': "🎯 Daily Challenge",
            'description': "Achieve 80% success rate in 50 episodes",
            'reward_xp': 100,
            'expires_in': "23:59:00"
        }


# Example usage
if __name__ == "__main__":
    system = GamificationSystem()
    
    # Simulate training stats
    stats = {
        'total_episodes': 150,
        'successes': 120,
        'success_rate': 0.8,
        'max_reward': 500,
        'min_steps': 75,
        'training_hours': 5
    }
    
    # Check achievements
    unlocked = system.check_achievements(stats)
    
    # Display progress
    progress = system.get_progress()
    print(f"\n📊 Your Progress:")
    print(f"   Level: {progress['level']}")
    print(f"   XP: {progress['xp']}")
    print(f"   Next Level: {progress['xp_to_next_level']} XP")
    print(f"   Achievements: {progress['achievements_unlocked']}/{progress['achievements_total']}")
    print(f"   Completion: {progress['completion_percentage']:.1f}%")
    
    # Show daily challenge
    challenge = system.get_daily_challenge()
    print(f"\n{challenge['title']}")
    print(f"   {challenge['description']}")
    print(f"   Reward: {challenge['reward_xp']} XP")
