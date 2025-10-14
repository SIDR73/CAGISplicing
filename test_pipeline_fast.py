#!/usr/bin/env python3
"""
Fast Test Runner for DrASNet Pipeline
Runs the pipeline in fast mode for quick testing and validation.
"""

import sys
import os
from complete_drasnet_pipeline import main

def run_fast_test():
    """
    Run the pipeline in fast mode for testing
    """
    print("🧪 FAST TEST MODE - DrASNet Pipeline")
    print("=" * 50)
    print("This will run the pipeline with smaller models for quick testing.")
    print("Use this to verify everything works before running the full pipeline.")
    print()
    
    try:
        # Run in fast mode
        predictions = main(fast_mode=True)
        
        print("\n" + "=" * 50)
        print("✅ FAST TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("The pipeline is working correctly.")
        print("You can now run the full pipeline with:")
        print("  python complete_drasnet_pipeline.py")
        print()
        print("Or use the runner script:")
        print("  python run_complete_pipeline.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during fast test: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_fast_test())
