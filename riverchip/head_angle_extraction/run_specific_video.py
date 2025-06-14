#!/usr/bin/env python3
"""
Runner script for head angle extraction on a specific video.
"""

import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_angle_extraction.main_pipeline import run_head_angle_extraction

def main():
    """Run head angle extraction on specific video."""
    
    # Specify the exact H5 file for the video you want to process
    input_file = "/home/lilly/phd/riverchip/data_analyzed/head_segmentation/data_original-MMH223_20250519_10_headsegmentation.h5"
    
    print(f"🎯 Running head angle extraction on: {input_file}")
    
    try:
        # Run the pipeline with the specific input file
        results_df = run_head_angle_extraction(input_file=input_file, use_parallel=True)
        
        print(f"\n📊 Final Results Summary:")
        print(f"   - Total measurements: {len(results_df)}")
        if len(results_df) > 0:
            print(f"   - Angle range: {results_df['angle_degrees'].min():.1f}° to {results_df['angle_degrees'].max():.1f}°")
            print(f"   - Mean angle: {results_df['angle_degrees'].mean():.1f}°")
            print(f"   - Straight frames: {sum(results_df['is_straight'])}")
            print(f"   - Bent frames: {sum(~results_df['is_straight'])}")
        
        print(f"\n✅ Processing complete! Results saved to:")
        print(f"   - CSV: /home/lilly/phd/riverchip/data_analyzed/final_data/data_original-MMH223_20250519_05_headangles.csv")
        print(f"   - Plots: /home/lilly/phd/riverchip/data_analyzed/plots/data_original-MMH223_20250519_05_head_angles.png")
        print(f"   - Video generation: Disabled for faster processing")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 