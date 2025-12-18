import os
from copy import deepcopy

def update_params(base, **overrides):
    """
    Return a new dict with updated parameters.
    Example:
        current = update_params(params_base, BG_NOISE=0.05, ANNEALING=2)
    """
    new_params = deepcopy(base)
    for key, value in overrides.items():
        # key format: "params.BG_NOISE" or "const.HVC_SIZE"
        section, param = key.split(".")
        new_params[section][param] = value
    return new_params


# save plots in a folder
save_dir = "plots"
def remove_prev_files(): # 
    '''removes previous files in the directory'''
    os.makedirs(save_dir, exist_ok = True)
    for filename in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, filename))

def find_neighboring_directories():
    """
    Finds all directories (folders, except pycache) in the same directory as the currently running Python script.

    Returns:
        list: A list of directory names found in the same directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_entries = os.listdir(current_dir)
    directories = []
    for entry in all_entries:
        if entry != "__pycache__":  # Skip the cache directory
            full_path = os.path.join(current_dir, entry)  # Get full path for entry
            if os.path.isdir(full_path):  # Check if it's a directory
                if entry != "__pycache__" and entry != "plots": 
                    directories.append(entry)
    return directories

