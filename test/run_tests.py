#!/usr/bin/env python3
"""
Simple script to run the CLI redaction test suite.

This script demonstrates how to run the comprehensive test suite
that covers all the examples from the CLI epilog.
"""

import os
import sys

# Add the parent directory to the path so we can import the test module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import run_all_tests

if __name__ == "__main__":
    print("Starting CLI Redaction Test Suite...")
    print("This will test all examples from the CLI epilog.")
    print("=" * 60)

    success = run_all_tests()

    if success:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)
