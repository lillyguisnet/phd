"""
Configuration settings for head angle extraction pipeline.
Centralized configuration for easy tuning and debugging.
"""

import multiprocessing as mp

class Config:
    """Configuration class for head angle extraction pipeline."""
    
    # Multiprocessing Configuration - OPTIMIZED FOR 48-CORE MACHINE
    USE_MULTIPROCESSING = True  # Set to False if multiprocessing causes issues
    MAX_PROCESSES = 46  # Use almost all cores (48 total, leave 2 for system)
    CHUNK_SIZE_MULTIPLIER = 4  # Increase chunks per process for better load balancing
    
    # Debug and Output Control Flags
    DEBUG_MODE = True
    SAVE_CSV = False  # Skip CSV saving for debugging
    GENERATE_VIDEO = False  # Skip video generation for debugging
    GENERATE_PLOTS = True  # Keep basic plots for visualization
    VERBOSE_TIMING = True  # Show detailed timing information
    
    # Data Paths
    HEAD_SEGMENTATION_DIR = "/home/lilly/phd/riverchip/data_analyzed/head_segmentation"
    FINAL_DATA_DIR = "/home/lilly/phd/riverchip/data_analyzed/final_data"
    VIDEO_DIR = "/home/lilly/phd/riverchip/data_foranalysis/videotojpg/data_original-hannah"
    
    # Algorithm Parameters
    MIN_VECTOR_LENGTH = 5
    RESTRICTION_POINT = 0.5  # Fixed: Original uses 0.5, not 0.4
    STRAIGHT_THRESHOLD = 3
    SMOOTHING_WINDOW = 3
    DEVIATION_THRESHOLD = 50
    
    # Truncation Parameters
    TAIL_TRIM_PIXELS = 10  # For conservative truncation
    KEEP_PIXELS = 400  # For fixed truncation (matches working version)
    
    # Video Generation Parameters
    VIDEO_FPS = 10
    BOTTOM_ALPHA = 0.3
    TOP_ALPHA = 0.7
    
    @classmethod
    def get_num_processes(cls):
        """Get the number of processes to use based on configuration."""
        if not cls.USE_MULTIPROCESSING:
            return 1
        return min(mp.cpu_count(), cls.MAX_PROCESSES)
    
    @classmethod
    def debug_print(cls, message):
        """Print debug messages only when DEBUG_MODE is True"""
        if cls.DEBUG_MODE:
            print(f"🐛 DEBUG: {message}")
    
    @classmethod
    def should_save_csv(cls):
        """Check if CSV saving is enabled."""
        return cls.SAVE_CSV
    
    @classmethod
    def should_generate_video(cls):
        """Check if video generation is enabled."""
        return cls.GENERATE_VIDEO
    
    @classmethod
    def should_generate_plots(cls):
        """Check if plot generation is enabled."""
        return cls.GENERATE_PLOTS 