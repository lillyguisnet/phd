"""
Main pipeline orchestration for head angle extraction.
Coordinates all the processing steps in the correct order.
"""

from .config import Config
from .data_loader import load_head_segments, get_random_unprocessed_video
from .parallel_processor import ParallelProcessor
from .skeleton_processor import generate_skeletons, truncate_skeletons
from .angle_calculator import calculate_head_angles
from .output_handler import save_results, generate_plots, generate_video

def run_head_angle_extraction(input_file=None, use_parallel=True):
    """
    Main pipeline for head angle extraction.
    
    Args:
        input_file: Path to H5 file or None for random selection
        use_parallel: Whether to use parallel processing
        
    Returns:
        DataFrame with results
    """
    print("🚀 Starting head angle extraction pipeline")
    
    # Initialize parallel processor
    processor = ParallelProcessor() if use_parallel else None
    
    # Load data
    if input_file is None:
        input_file = get_random_unprocessed_video()
        print(f"📁 Selected random video: {input_file}")
    
    head_segments = load_head_segments(input_file)
    
    # Generate skeletons
    skeletons, skeleton_stats = generate_skeletons(head_segments, processor)
    
    # Truncate skeletons
    Config.debug_print("Truncating skeletons using fixed method (400 pixels from top)")
    truncated_skeletons = truncate_skeletons(
        skeletons, 
        method="fixed",  # Use fixed method like working version
        keep_pixels=400,  # Keep 400 pixels from top like working version
        processor=processor
    )
    
    # Calculate head angles
    results_df = calculate_head_angles(truncated_skeletons)
    
    # Generate outputs
    save_results(results_df, input_file)
    generate_plots(results_df)
    generate_video(head_segments, skeletons, results_df)
    
    print("🎉 Head angle extraction pipeline completed!")
    return results_df

if __name__ == "__main__":
    # Allow running the pipeline directly
    result = run_head_angle_extraction()
    print("Pipeline completed successfully!") 