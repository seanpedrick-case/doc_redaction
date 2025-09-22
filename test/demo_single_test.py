#!/usr/bin/env python3
"""
Demonstration script showing how to run a single test example.

This script shows how to use the run_cli_redact function directly
to test a specific CLI example.
"""

import os
import sys
import tempfile
import shutil

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.test import run_cli_redact

def demo_pdf_redaction():
    """Demonstrate how to run a single PDF redaction test."""
    print("=== Demo: PDF Redaction with Default Settings ===")
    
    # Set up paths
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cli_redact.py")
    input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example_data", "example_of_emails_sent_to_a_professor_before_applying.pdf")
    output_dir = tempfile.mkdtemp(prefix="demo_output_")
    
    print(f"Script: {script_path}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    
    # Check if files exist
    if not os.path.isfile(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    if not os.path.isfile(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Make sure you have the example data files in the example_data/ directory")
        return False
    
    try:
        # Run the test
        print("\nRunning PDF redaction with default settings...")
        result = run_cli_redact(
            script_path=script_path,
            input_file=input_file,
            output_dir=output_dir
        )
        
        if result:
            print("✅ Test completed successfully!")
            print(f"Check the output directory for results: {output_dir}")
        else:
            print("❌ Test failed!")
        
        return result
        
    finally:
        # Clean up
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleaned up: {output_dir}")

def demo_csv_anonymisation():
    """Demonstrate how to run a CSV anonymisation test."""
    print("\n=== Demo: CSV Anonymisation ===")
    
    # Set up paths
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cli_redact.py")
    input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example_data", "combined_case_notes.csv")
    output_dir = tempfile.mkdtemp(prefix="demo_output_")
    
    print(f"Script: {script_path}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    
    # Check if files exist
    if not os.path.isfile(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    if not os.path.isfile(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Make sure you have the example data files in the example_data/ directory")
        return False
    
    try:
        # Run the test
        print("\nRunning CSV anonymisation...")
        result = run_cli_redact(
            script_path=script_path,
            input_file=input_file,
            output_dir=output_dir,
            text_columns=["Case Note", "Client"],
            anon_strategy="replace_redacted"
        )
        
        if result:
            print("✅ Test completed successfully!")
            print(f"Check the output directory for results: {output_dir}")
        else:
            print("❌ Test failed!")
        
        return result
        
    finally:
        # Clean up
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleaned up: {output_dir}")

if __name__ == "__main__":
    print("CLI Redaction Test Demo")
    print("=" * 50)
    print("This script demonstrates how to run individual tests.")
    print("=" * 50)
    
    # Run the demos
    success1 = demo_pdf_redaction()
    success2 = demo_csv_anonymisation()
    
    print("\n" + "=" * 50)
    print("Demo Summary")
    print("=" * 50)
    print(f"PDF Redaction: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"CSV Anonymisation: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    overall_success = success1 and success2
    print(f"\nOverall: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    sys.exit(0 if overall_success else 1)
