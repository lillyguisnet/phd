#!/usr/bin/env python3
"""
Runner script for head angle extraction pipeline.
Simple entry point to execute the complete pipeline.
"""

import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_angle_extraction.main_pipeline import run_head_angle_extraction

def main():
    """Main entry point for the head angle extraction pipeline."""
    try:
        # Run the pipeline
        results_df = run_head_angle_extraction()
        
        print(f"\n📊 Final Results Summary:")
        print(f"   - Total measurements: {len(results_df)}")
        if len(results_df) > 0:
            print(f"   - Angle range: {results_df['angle_degrees'].min():.1f}° to {results_df['angle_degrees'].max():.1f}°")
            print(f"   - Mean angle: {results_df['angle_degrees'].mean():.1f}°")
            print(f"   - Straight frames: {sum(results_df['is_straight'])}")
            print(f"   - Bent frames: {sum(~results_df['is_straight'])}")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 