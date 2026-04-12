"""
Performance and energy tracking utilities for benchmarking.
Monitors CPU usage, memory consumption, and energy footprint.

Note: codecarbon is imported lazily inside BenchmarkTracker.start_benchmark()
so that a missing installation degrades gracefully (energy_kwh = 0.0) instead
of crashing the entire import chain.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    avg_fps: float
    avg_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float
    total_energy_kwh: float
    total_inference_time: float
    total_frames_processed: int


class ResourceMonitor:
    """
    Real-time resource monitoring for CPU and memory usage.
    """
    
    def __init__(self):
        self.monitoring = False
        self.cpu_readings: List[float] = []
        self.memory_readings: List[float] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.stop_event.clear()
        self.cpu_readings.clear()
        self.memory_readings.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background resource monitoring."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        # Initialize psutil
        psutil.cpu_percent(interval=None)
        
        while not self.stop_event.is_set():
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 * 1024)
                
                self.cpu_readings.append(cpu_percent)
                self.memory_readings.append(memory_mb)
                
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
                
    def get_average_cpu(self) -> float:
        """Get average CPU usage during monitoring."""
        if not self.cpu_readings:
            return 0.0
        return np.mean(self.cpu_readings)
        
    def get_average_memory(self) -> float:
        """Get average memory usage during monitoring."""
        if not self.memory_readings:
            return 0.0
        return np.mean(self.memory_readings)
        
    def get_peak_memory(self) -> float:
        """Get peak memory usage during monitoring."""
        if not self.memory_readings:
            return 0.0
        return np.max(self.memory_readings)


class BenchmarkTracker:
    """
    Comprehensive benchmark tracking with energy and performance metrics.
    """
    
    def __init__(self, project_name: str = "digit_benchmark"):
        self.project_name = project_name
        self.resource_monitor = ResourceMonitor()
        self.emissions_tracker = None
        self.start_time = None
        self.end_time = None
        self.frames_processed = 0
        
    def start_benchmark(self):
        """Start comprehensive benchmark tracking."""
        self.start_time = time.time()
        self.frames_processed = 0

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

        # Start energy tracking — lazy import so a missing codecarbon
        # installation does not crash the import chain.
        try:
            from codecarbon import EmissionsTracker  # lazy import
            self.emissions_tracker = EmissionsTracker(
                project_name=self.project_name,
                log_level="error",
                save_to_file=False,
            )
            self.emissions_tracker.start()
        except ImportError:
            # codecarbon not installed — energy tracking silently disabled.
            self.emissions_tracker = None
        except Exception as e:
            print(f"Warning: Could not start energy tracking: {e}")
            self.emissions_tracker = None
            
    def stop_benchmark(self) -> PerformanceMetrics:
        """Stop benchmark tracking and return metrics."""
        self.end_time = time.time()
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Stop energy tracking
        energy_kwh = 0.0
        if self.emissions_tracker:
            try:
                emissions_data = self.emissions_tracker.stop()
                if hasattr(emissions_data, 'energy_consumed'):
                    energy_kwh = emissions_data.energy_consumed
                elif hasattr(self.emissions_tracker, '_total_energy'):
                    energy_kwh = self.emissions_tracker._total_energy.kWh
            except Exception as e:
                print(f"Warning: Error stopping energy tracker: {e}")
                
        # Calculate metrics
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        avg_fps = self.frames_processed / total_time if total_time > 0 else 0
        
        return PerformanceMetrics(
            avg_fps=avg_fps,
            avg_cpu_percent=self.resource_monitor.get_average_cpu(),
            avg_memory_mb=self.resource_monitor.get_average_memory(),
            peak_memory_mb=self.resource_monitor.get_peak_memory(),
            total_energy_kwh=energy_kwh,
            total_inference_time=total_time,
            total_frames_processed=self.frames_processed
        )
        
    def increment_frame_count(self, count: int = 1):
        """Increment the frame counter."""
        self.frames_processed += count
        
    def get_current_metrics(self) -> Dict:
        """Get current benchmark metrics (partial)."""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        current_fps = self.frames_processed / elapsed if elapsed > 0 else 0
        
        return {
            'elapsed_time': elapsed,
            'frames_processed': self.frames_processed,
            'current_fps': current_fps,
            'current_cpu': self.resource_monitor.get_average_cpu(),
            'current_memory_mb': self.resource_monitor.get_average_memory()
        }



