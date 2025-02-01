"""
Training progress visualization and metrics tracking.
"""

import sys
import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingMetrics:
    """Training metrics container"""
    loss: float
    learning_rate: float
    speed: float  # iterations per second
    gpu_memory: Optional[float] = None
    epoch: int = 0
    total_epochs: int = 0
    iteration: int = 0
    total_iterations: int = 0

class ProgressTracker:
    """Tracks and displays training progress with rich formatting"""
    
    def __init__(self, stage: str, total_epochs: int, total_iterations: int):
        self.stage = stage
        self.total_epochs = total_epochs
        self.total_iterations = total_iterations
        self.start_time = time.time()
        self.last_update = self.start_time
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
    def _get_eta(self, progress: float) -> str:
        """Calculate estimated time remaining"""
        if progress == 0:
            return "--:--:--"
        elapsed = time.time() - self.start_time
        eta = (elapsed / progress) * (1 - progress)
        return self._format_time(eta)
        
    def _format_progress_bar(self, progress: float, width: int = 30) -> str:
        """Create a progress bar string"""
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return bar
        
    def update(self, metrics: TrainingMetrics):
        """Update and display training progress"""
        # Move cursor up to clear previous output
        if hasattr(self, 'last_lines'):
            sys.stdout.write(f"\033[{self.last_lines}A")  # Move up
            sys.stdout.write("\033[J")  # Clear to end of screen
        
        # Calculate progress
        epoch_progress = metrics.epoch / self.total_epochs
        total_progress = (metrics.epoch * self.total_iterations + metrics.iteration) / (self.total_epochs * self.total_iterations)
        
        # Calculate speed and time estimates
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Create status lines
        lines = []
        lines.append(f"{self.stage} Training Progress:")
        
        # Epoch progress
        lines.append(f"Epoch [{metrics.epoch}/{self.total_epochs}] {self._format_progress_bar(epoch_progress)} {epoch_progress*100:.1f}%")
        
        # Overall progress
        lines.append(f"Total Progress: {self._format_progress_bar(total_progress)} {total_progress*100:.1f}%")
        
        # Metrics
        metric_line = f"Loss: {metrics.loss:.6f} | LR: {metrics.learning_rate:.2e} | Speed: {metrics.speed:.2f} it/s"
        if metrics.gpu_memory:
            metric_line += f" | GPU Memory: {metrics.gpu_memory:.1f}GB"
        lines.append(metric_line)
        
        # Time information
        lines.append(f"Elapsed: {self._format_time(elapsed)} | ETA: {self._get_eta(total_progress)}")
        
        # Add warning indicators
        warnings = []
        if metrics.speed < 0.15:
            warnings.append("⚠️  Training speed is low")
        if metrics.loss == 0.0:
            warnings.append("⚠️  Loss is zero - potential underfit")
        if metrics.learning_rate < 1e-6:
            warnings.append("⚠️  Learning rate is very low")
        
        if warnings:
            lines.extend(warnings)
        
        # Join lines and print
        output = "\n".join(lines)
        print(output)
        
        # Store number of lines for next update
        self.last_lines = len(lines)
        
        # Update last update time
        self.last_update = current_time

def create_stage_header(stage: str) -> str:
    """Create a formatted header for training stages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n{'='*80}\n"
    header += f"Starting {stage} Training Stage at {timestamp}\n"
    header += f"{'='*80}\n"
    return header

def create_stage_summary(
    stage: str,
    best_loss: float,
    final_speed: float,
    checkpoint_path: str
) -> str:
    """Create a formatted summary for completed stages"""
    summary = f"\n{'-'*80}\n"
    summary += f"{stage} Training Complete:\n"
    summary += f"Best Loss: {best_loss:.6f}\n"
    summary += f"Final Training Speed: {final_speed:.2f} it/s\n"
    summary += f"Checkpoint Saved: {checkpoint_path}\n"
    summary += f"{'-'*80}\n"
    return summary
