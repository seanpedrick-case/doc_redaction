# get_cdk_template.py
import subprocess
import json
import os
import shutil

from cdk_config import CONTEXT_FILE, CDK_CONFIG_PATH

# Synth your CDK app to local files and print out the full logs to file

cdk_folder = '' #<FULL_PATH_TO_CDK_FOLDER_HERE>

#print(os.environ["CONTEXT_FILE"])

# Full path needed to find config file
os.environ["CDK_CONFIG_PATH"] = cdk_folder + CDK_CONFIG_PATH

print(os.environ["CONTEXT_FILE"])
print(os.environ["CDK_CONFIG_PATH"])

def synthesize_cdk_stack_to_json(stack_name: str) -> dict:
    cdk_executable = shutil.which('cdk')
    if not cdk_executable:
        raise FileNotFoundError(
            "The 'cdk' command was not found in your system's PATH. "
            "Please ensure AWS CDK CLI is installed and accessible."
        )

    # Calculate the CDK project root based on the script's location
    # Your script (get_cdk_template.py) is at:
    # C:\Users\spedrickcase\OneDrive - Lambeth Council\Apps\doc_redaction\cdk\get_cdk_template.py
    # Your project root (where cdk.json is) is at:
    # C:\Users\spedrickcase\OneDrive - Lambeth Council\Apps\doc_redaction\

    # Get the directory of the current script:
    # C:\Users\spedrickcase\OneDrive - Lambeth Council\Apps\doc_redaction\cdk\
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one more level to reach the actual project root (doc_redaction\)
    # This is the directory where cdk.json should be located
    cdk_project_root = os.path.dirname(current_script_dir)

    print(f"Calculated CDK project root for subprocess: {cdk_project_root}")

    command = [cdk_executable, 'synth', stack_name]
    print(f"Attempting to run command: {' '.join(command)} from directory: {cdk_project_root}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=cdk_project_root # This must be a directory path
        )

        cloudformation_template_json_str = result.stdout

        print("cloudformation_template_json_str:", cloudformation_template_json_str)
        # ... (rest of your JSON parsing and error handling logic) ...
        if not cloudformation_template_json_str.strip().startswith('{'):
            # This part should be improved if you face actual non-JSON output
            # For now, it just re-raises, but for robustness you might want to log/clean before json.loads
            #raise json.JSONDecodeError("Output from 'cdk synth' is not valid JSON.", cloudformation_template_json_str, 0)
            print("Output from 'cdk synth' is not valid JSON.", cloudformation_template_json_str, 0)

            # Return string
            return cloudformation_template_json_str, 'str'
        else:
            cf_template_dict = json.loads(cloudformation_template_json_str)
            return cf_template_dict, 'json'

    except FileNotFoundError:
        print(f"Error: Command '{cdk_executable}' not found. Verify its path and permissions.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error synthesizing stack '{stack_name}':")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}") # This might contain CDK error messages
        print(f"STDERR:\n{e.stderr}") # This might contain CDK error messages
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON output from 'cdk synth': {e}")
        print(f"Raw output (first 500 chars):\n{cloudformation_template_json_str[:500]}...")
        print(f"Full STDOUT from cdk synth:\n{cloudformation_template_json_str}")
        print(f"Full STDERR from cdk synth:\n{result.stderr}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    stack_to_synthesize = "RedactionStack"

    # The stray print(__file__) from earlier in the traceback is gone here.
    # If it's still present in your get_cdk_template.py, remove it.

    print(f"Synthesizing stack '{stack_to_synthesize}'...")
    try:
        template_dict, type = synthesize_cdk_stack_to_json(stack_to_synthesize)

        print(f"\nSuccessfully obtained CloudFormation template for '{stack_to_synthesize}'.")

        # Example: Print a snippet of the template
        if type == "json":
            print("\n--- First 200 characters of the JSON template ---")
            print(json.dumps(template_dict, indent=2)[:200] + "...")

            # Example: Save the template to a file
            output_filename = f"{stack_to_synthesize}.template.json"
            # Ensure the output file is saved in the current working directory,
            # which is where you run this script (likely the root of your CDK project).
            with open(os.path.join(os.getcwd(), output_filename), 'w') as f:
                json.dump(template_dict, f, indent=2)
            print(f"\nCloudFormation template saved to: {os.path.join(os.getcwd(), output_filename)}")
        else:
            print("First 200 characters of output:", template_dict[0:200])

            output_filename = f"{stack_to_synthesize}.template.txt"
            # Ensure the output file is saved in the current working directory,
            # which is where you run this script (likely the root of your CDK project).
            with open(os.path.join(os.getcwd(), output_filename), 'w') as f:
                f.write(template_dict)
            print(f"\nCloudFormation template saved to: {os.path.join(os.getcwd(), output_filename)}")


    except Exception as e:
        print(f"\nFailed to synthesize stack: {e}")