import json
import os

def find_neighboring_directories():
    """
    Finds all directories (folders) in the same directory as the currently running Python script.

    Returns:
        list: A list of directory names found in the same directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_entries = os.listdir(current_dir)
    directories = []
    for entry in all_entries:
        full_path = os.path.join(current_dir, entry)  # Get full path for entry
        if os.path.isdir(full_path):  # Check if it's a directory
            if not entry.startswith("__pycache__"):  # Skip Python cache directories
                directories.append(entry)
    return directories

def modify_json(filename, parameter_path, new_value, new_filename="modified_params.json"):
    """
    Opens a JSON file, modifies a specific parameter value, and saves the changes to a new file.

    Args:
        filename (str): Path to the original JSON file.
        parameter_path (str): A string representing the path to the parameter within the JSON structure (e.g., "modes/ANNEALING").
        new_value: The new value to assign to the parameter.
        new_filename (str, optional): Path to the new file where the modified data will be saved. Defaults to "modified_params.json".
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # Access and modify the parameter
    keys = parameter_path.split("/")
    current_dict = data
    for key in keys[:-1]:
        current_dict = current_dict[key]
    current_dict[keys[-1]] = new_value

    # Save to a new file
    with open(new_filename, "w") as f:
        json.dump(data, f, indent=2)  # Save with indentation for readability

# Neighbour directories
neighboring_directories = find_neighboring_directories()

print("Neighboring files:")
for directory in neighboring_directories:
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path) and filename.endswith(".json"):
            os.remove(full_path)
            print(f"Removed JSON file: {full_path}")

# Define parameter values
# Define parameter values
BG_NOISE_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
DECAY_FACTOR_values = [1, 2, 2.2, 2.5, 3]


# Define parameter names and corresponding values
parameter_names = {
    "BG_NOISE": BG_NOISE_values,
    "DECAY_FACTOR": DECAY_FACTOR_values
}

filename = "params.json"
for directory in neighboring_directories:
    print(f"Directory: {directory}")    
    if directory in parameter_names:
        parameter_name = directory
        parameter_values = parameter_names[directory]
    else:
        print(f"Skipping directory '{directory}' as it's not in the parameter_names list.")

    for value in parameter_values:
        new_filename = f"{directory}/parameters_{value}.json"
        parameter_path = f"params/{parameter_name}"  # Modify this if your structure is different
        new_value = value
        modify_json(filename, parameter_path, new_value, new_filename)
        print(f"Modified parameter '{parameter_path}' to {new_value} and saved to {new_filename}")