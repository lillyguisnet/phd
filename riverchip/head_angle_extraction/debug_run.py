#!/usr/bin/env python3
"""
Debug script to run head angle extraction with detailed error handling.
"""

import sys
import os
import traceback

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_angle_extraction.config import Config
from head_angle_extraction.data_loader import load_head_segments
from head_angle_extraction.parallel_processor import ParallelProcessor
from head_angle_extraction.skeleton_processor import generate_skeletons, truncate_skeletons
from head_angle_extraction.angle_calculator import calculate_head_angles
from head_angle_extraction.output_handler import save_results, generate_plots, generate_video

def debug_run_pipeline():
    """Run the pipeline with detailed debugging."""
    
    input_file = "/home/lilly/phd/riverchip/data_analyzed/head_segmentation/data_original-MMH223_20250519_05_headsegmentation.h5"
    
    print("🚀 Starting debug head angle extraction pipeline")
    print(f"📁 Input file: {input_file}")
    
    try:
        # Step 1: Initialize parallel processor
        print("\n📊 Step 1: Initializing parallel processor...")
        processor = ParallelProcessor()
        print("✅ Parallel processor initialized")
        
        # Step 2: Load data
        print("\n📊 Step 2: Loading head segments...")
        head_segments = load_head_segments(input_file)
        print(f"✅ Head segments loaded: {len(head_segments)} frames")
        
        # Step 3: Generate skeletons
        print("\n📊 Step 3: Generating skeletons...")
        skeletons, skeleton_stats = generate_skeletons(head_segments, processor)
        print(f"✅ Skeletons generated: {len(skeletons)} frames")
        
        # Step 4: Truncate skeletons
        print("\n📊 Step 4: Truncating skeletons...")
        Config.debug_print("Truncating skeletons using fixed method (400 pixels from top)")
        truncated_skeletons = truncate_skeletons(
            skeletons, 
            method="fixed",
            keep_pixels=400,
            processor=processor
        )
        print(f"✅ Skeletons truncated: {len(truncated_skeletons)} frames")
        
        # Step 5: Calculate head angles
        print("\n📊 Step 5: Calculating head angles...")
        results_df = calculate_head_angles(truncated_skeletons)
        print(f"✅ Head angles calculated: {len(results_df)} measurements")
        
        # Step 6: Save results
        print("\n📊 Step 6: Saving results...")
        results_with_smoothing = save_results(results_df, input_file)
        print("✅ Results saved to CSV")
        
        # Step 7: Generate plots
        print("\n📊 Step 7: Generating plots...")
        generate_plots(results_with_smoothing, input_file)
        print("✅ Plots generated")
        
        # Step 8: Generate video (if enabled)
        print("\n📊 Step 8: Checking video generation...")
        if Config.should_generate_video():
            print("Generating video...")
            generate_video(head_segments, skeletons, results_df)
            print("✅ Video generated")
        else:
            print("⏭️  Video generation disabled")
        
        print("\n🎉 Pipeline completed successfully!")
        
        # Final summary
        print(f"\n📊 Final Results Summary:")
        print(f"   - Total measurements: {len(results_df)}")
        if len(results_df) > 0:
            print(f"   - Angle range: {results_df['angle_degrees'].min():.1f}° to {results_df['angle_degrees'].max():.1f}°")
            print(f"   - Mean angle: {results_df['angle_degrees'].mean():.1f}°")
            print(f"   - Straight frames: {sum(results_df['is_straight'])}")
            print(f"   - Bent frames: {sum(~results_df['is_straight'])}")
        
        return results_df
        
    except Exception as e:
        print(f"\n❌ ERROR in pipeline at step: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = debug_run_pipeline()
    if result is not None:
        print(f"\n✅ SUCCESS: Pipeline completed with {len(result)} results")
    else:
        print("\n❌ FAILED: Pipeline did not complete") 