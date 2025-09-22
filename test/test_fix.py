#!/usr/bin/env python3
"""
Quick test script to verify the fix for the None value error.
"""

import os
import sys
import tempfile
import shutil

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import run_cli_redact

def test_none_handling():
    """Test that None values are handled properly."""
    print("Testing None value handling...")
    
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cli_redact.py")
    output_dir = tempfile.mkdtemp(prefix="test_fix_")
    
    try:
        # Test with None input file (should handle gracefully for list action)
        result = run_cli_redact(
            script_path=script_path,
            input_file=None,  # No input file needed for list action
            output_dir=output_dir,
            task="textract",
            textract_action="list"
        )
        
        print("✅ None value handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ None value handling test failed: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    print("Testing the fix for None value error...")
    success = test_none_handling()
    
    if success:
        print("\n🎉 Fix appears to be working!")
        print("You should now be able to run the full test suite without the None value error.")
    else:
        print("\n❌ Fix needs more work.")
    
    sys.exit(0 if success else 1)
