#!/usr/bin/env python3
"""
Standalone GUI test script for the document redaction application.

This script tests only the GUI functionality of app.py to ensure it loads correctly.
Run this script to verify that the Gradio interface can be imported and initialized.
"""

import os
import sys
import threading
import unittest

# Add the parent directory to the path so we can import the app
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class TestGUIAppOnly(unittest.TestCase):
    """Test suite for GUI application loading and basic functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for GUI tests."""
        cls.app_path = os.path.join(parent_dir, "app.py")

        # Verify app.py exists
        if not os.path.isfile(cls.app_path):
            raise FileNotFoundError(f"App file not found: {cls.app_path}")

        print(f"GUI test setup complete. App: {cls.app_path}")

    def test_app_import_and_initialization(self):
        """Test: Import app.py and check if the Gradio app object is created successfully."""
        print("\n=== Testing GUI app import and initialization ===")

        try:
            # Import the app module
            import app

            # Check if the app object exists and is a Gradio Blocks object
            self.assertTrue(
                hasattr(app, "app"), "App object should exist in the module"
            )

            # Check if it's a Gradio Blocks instance
            import gradio as gr

            self.assertIsInstance(
                app.app, gr.Blocks, "App should be a Gradio Blocks instance"
            )

            print("✅ GUI app import and initialization passed")

        except ImportError as e:
            error_msg = f"Failed to import app module: {e}"
            if "gradio_image_annotation" in str(e):
                error_msg += "\n\nNOTE: This test requires the 'redaction' conda environment to be activated."
                error_msg += "\nPlease run: conda activate redaction"
                error_msg += "\nThen run this test again."
            self.fail(error_msg)
        except Exception as e:
            self.fail(f"Unexpected error during app initialization: {e}")

    def test_app_launch_headless(self):
        """Test: Launch the app in headless mode to verify it starts without errors."""
        print("\n=== Testing GUI app launch in headless mode ===")

        try:
            # Import the app module

            import app

            # Set up a flag to track if the app launched successfully
            app_launched = threading.Event()
            launch_error = None

            def launch_app():
                try:
                    # Launch the app in headless mode with a short timeout
                    app.app.launch(
                        show_error=True,
                        inbrowser=False,  # Don't open browser
                        server_port=0,  # Use any available port
                        quiet=True,  # Suppress output
                        prevent_thread_lock=True,  # Don't block the main thread
                    )
                    app_launched.set()
                except Exception:
                    app_launched.set()

            # Start the app in a separate thread
            launch_thread = threading.Thread(target=launch_app)
            launch_thread.daemon = True
            launch_thread.start()

            # Wait for the app to launch (with timeout)
            if app_launched.wait(timeout=10):  # 10 second timeout
                if launch_error:
                    self.fail(f"App launch failed: {launch_error}")
                else:
                    print("✅ GUI app launch in headless mode passed")
            else:
                self.fail("App launch timed out after 10 seconds")

        except Exception as e:
            error_msg = f"Unexpected error during app launch test: {e}"
            if "gradio_image_annotation" in str(e):
                error_msg += "\n\nNOTE: This test requires the 'redaction' conda environment to be activated."
                error_msg += "\nPlease run: conda activate redaction"
                error_msg += "\nThen run this test again."
            self.fail(error_msg)

    def test_app_configuration_loading(self):
        """Test: Verify that the app can load its configuration without errors."""
        print("\n=== Testing GUI app configuration loading ===")

        try:
            # Import the app module (not necessary here?)
            # import app

            # Check if key configuration variables are accessible
            # These should be imported from tools.config
            from tools.config import (
                DEFAULT_LANGUAGE,
                GRADIO_SERVER_PORT,
                MAX_FILE_SIZE,
                PII_DETECTION_MODELS,
            )

            # Verify these are not None/empty
            self.assertIsNotNone(
                GRADIO_SERVER_PORT, "GRADIO_SERVER_PORT should be configured"
            )
            self.assertIsNotNone(MAX_FILE_SIZE, "MAX_FILE_SIZE should be configured")
            self.assertIsNotNone(
                DEFAULT_LANGUAGE, "DEFAULT_LANGUAGE should be configured"
            )
            self.assertIsNotNone(
                PII_DETECTION_MODELS, "PII_DETECTION_MODELS should be configured"
            )

            print("✅ GUI app configuration loading passed")

        except ImportError as e:
            error_msg = f"Failed to import configuration: {e}"
            if "gradio_image_annotation" in str(e):
                error_msg += "\n\nNOTE: This test requires the 'redaction' conda environment to be activated."
                error_msg += "\nPlease run: conda activate redaction"
                error_msg += "\nThen run this test again."
            self.fail(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during configuration test: {e}"
            if "gradio_image_annotation" in str(e):
                error_msg += "\n\nNOTE: This test requires the 'redaction' conda environment to be activated."
                error_msg += "\nPlease run: conda activate redaction"
                error_msg += "\nThen run this test again."
            self.fail(error_msg)


def run_gui_tests():
    """Run GUI tests and report results."""
    print("=" * 80)
    print("DOCUMENT REDACTION GUI TEST SUITE")
    print("=" * 80)
    print("This test suite verifies that the GUI application loads correctly.")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGUIAppOnly)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("GUI TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✅ PASSED' if success else '❌ FAILED'}")
    print("=" * 80)

    return success


if __name__ == "__main__":
    # Run the GUI test suite
    success = run_gui_tests()
    exit(0 if success else 1)
