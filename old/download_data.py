import getpass
import logging
import os
from typing import List

import fire
from dotenv import load_dotenv
from radiant_mlhub import Dataset

from aic.helpers import get_dir

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s:', datefmt='%H:%M:%S')

MLHUB_API_KEY = os.getenv('MLHUB_API_KEY')

def main(
    dataset_name: str = 'ref_agrifieldnet_competition_v1',
    output_dir: str = 'data/source',
    selected_bands: List[str] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']):
    """
    Donwnload the dataset

    Args:
        dataset_name (str, optional): name of the dataset. Defaults to 'ref_agrifieldnet_competition_v1'.
        output_dir (str, optional): directories where to store the triaining data. Defaults to 'data/train_test'.
        selected_bands (List[str], optional): number of bands to use. Defaults to empty
    """

    source_collection = f'{dataset_name}_source'
    train_label_collection = f'{dataset_name}_labels_train'
    test_label_collection = f'{dataset_name}_labels_test'

    if not MLHUB_API_KEY:
        logging.warning('MLHub Api Key not found. Do you save it as an environmental variable?')
        os.environ['MLHUB_API_KEY'] =  getpass.getpass(prompt="MLHub API Key: ")

    # fetch informatin about the dataset
    dataset = Dataset.fetch(dataset_name)

    # handle filters
    assets = ['field_ids','raster_labels']
    filters = {
        train_label_collection: assets,
        test_label_collection: [assets[0]],
        source_collection: selected_bands
    }

    # download the dataset
    dataset.download(
        output_dir=get_dir(output_dir),
        collection_filter=filters
    )

if __name__ == "__main__":
    fire.Fire(main)
