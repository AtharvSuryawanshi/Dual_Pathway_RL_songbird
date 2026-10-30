import json

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

# Example usage
filename = "params.json"
parameter_path = "modes/ANNEALING"
new_value = False

modify_json(filename, parameter_path, new_value)
print(f"Modified parameter '{parameter_path}' to {new_value} and saved to {new_filename}")