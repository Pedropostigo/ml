"""
Module to handle JSON files in python: open file and get a dict, save dict in JSON file, etc.

Functions:
read_json - open a JSON file and get the information in a python dict format
save_json - save a python dict in JSON format
"""

import json

def read_json(file_path):
    """
    open a JSON file and return its information in a python dict format

    Parameters:
    file_path       -- path to the JSON file to open

    Return:
    json_to_dict    -- python dict containing the JSON information
    """

    # open the JSON file and save info into dict
    with open(file_path, 'r') as fp:
        json_to_dict = fp.read()
        json_to_dict = json.loads(json_to_dict)

    return json_to_dict


def save_json(dict_to_json, file_path):
    """
    save the information in a python dictionary to a JSON file

    Parameters:
    dict_to_json    -- dict containing the information to be saved to json
    file_path       -- path to the JSON file where the dict will be saved

    Return:
    None
    """
    with open(file_path, 'w') as fp:
        json.dump(dict_to_json, fp)