import os
import json

def get_dir(*paths) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name
    Returns:
        str:
    """
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return directory

def get_folder_ids(data_dir, dataset_name, collection):
    """
    Get folder id for each chip in the dataset in either train or test test

    Args:
        data_dir: data directory where the folders for each chip are found
        dataset_name: name of the dataset to use
        collection: collection to use
    """
    with open (f'{data_dir}/{dataset_name}/{collection}/collection.json') as f:
        collention_json = json.load(f)
        collention_folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in collention_json['links'][4:]]

        return collention_folder_ids