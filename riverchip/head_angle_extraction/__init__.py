"""
Head Angle Extraction Package

A modular package for extracting head angles from worm skeleton data.
Optimized for high-performance parallel processing on multi-core systems.
"""

from .config import Config
from .data_loader import load_head_segments, get_unprocessed_videos
from .skeleton_processor import generate_skeletons, truncate_skeletons  
from .angle_calculator import calculate_head_angles
from .parallel_processor import ParallelProcessor
from .output_handler import save_results, generate_plots, generate_video
from .main_pipeline import run_head_angle_extraction

__version__ = "1.0.0"
__all__ = [
    "Config",
    "load_head_segments", 
    "get_unprocessed_videos",
    "generate_skeletons",
    "truncate_skeletons", 
    "calculate_head_angles",
    "ParallelProcessor",
    "save_results",
    "generate_plots", 
    "generate_video",
    "run_head_angle_extraction"
] 