import json
import os
from functions import find_neighboring_directories, modify_json

# Neighbour directories
neighboring_directories = find_neighboring_directories()


print("Neighboring files:")
for directory in neighboring_directories:
    print(directory)
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path) and filename.endswith(".json"):
            os.remove(full_path)
            print(f"Removed JSON file: {full_path}")

# Define parameter values
BG_NOISE_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
DECAY_FACTOR_values = [1, 2.5, 5, 10]


# Define parameter names and corresponding values
parameter_names = {
    "BG_NOISE": BG_NOISE_values,
    "DECAY_FACTOR": DECAY_FACTOR_values
}

filename = "params.json"
for directory in neighboring_directories:
    if directory != "__pycache__":
        if directory in parameter_names:
            parameter_name = directory
            parameter_values = parameter_names[directory]

        for value in parameter_values:
            new_filename = f"{directory}/parameters_{value}.json"
            parameter_path = f"params/{parameter_name}"  # Modify this if your structure is different
            new_value = value
            modify_json(filename, parameter_path, new_value, new_filename)
            print(f"Modified parameter '{parameter_path}' to {new_value} and saved to {new_filename}")