import logging
from typing import List

import fire
from dotenv import load_dotenv
import json
from tqdm import tqdm
import rasterio
import numpy as np
from collections import defaultdict
import pandas as pd

from src.helpers import get_dir

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s:', datefmt='%H:%M:%S')

crops = {
    "1": "Wheat",
    "2": "Mustard",
    "3": "Lentil",
    "4": "No Crop/Fallow",
    "5": "Green pea",
    "6": "Sugarcane",
    "8": "Garlic",
    "9": "Maize",
    "13": "Gram",
    "14": "Coriander",
    "15": "Potato",
    "16": "Bersem",
    "36": "Rice",
}

def get_folder_ids(data_dir, dataset_name, collection):
    with open (f'{data_dir}/{dataset_name}/{collection}/collection.json') as f:
        collention_json = json.load(f)
        collention_folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in collention_json['links'][4:]]

        return collention_folder_ids

def main(
    format: str,
    dataset_name: str = 'ref_agrifieldnet_competition_v1',
    data_dir: str = 'data/train_test',
    selected_bands: List[str] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']):
    """
    Preprocess data for segmentation. Expects a SampleSubmission file

    Args:
        format (str):  format in which the preprocessed data will be used
        dataset_name (str, optional): name of the dataset. Defaults to 'ref_agrifieldnet_competition_v1'.
        data_dir (str, optional): directories where to store the triaining data. Defaults to 'data/train_test'.
        selected_bands (List[str], optional): number of bands to use. Defaults to empty
    """


    source_collection = f'{dataset_name}_source'
    test_label_collection = f'{dataset_name}_labels_test'

    if format == "tabular":

        logging.info("Preprocessing data for a tabular testing...")

        def get_all_band_pixel_for_field_id(folder_ids):

            pbar = tqdm(folder_ids)

            data = []

            for idx in pbar:

                with rasterio.open(f'{data_dir}/{dataset_name}/{test_label_collection}/{test_label_collection}_{idx}/field_ids.tif') as src:
                    field_data = src.read()[0]

                # get bands for folder id
                bands_src = [rasterio.open(f'{data_dir}/{dataset_name}/{source_collection}/{source_collection}_{idx}/{band}.tif') for band in selected_bands]
                bands_array = [np.expand_dims(band.read(1), axis=0) for band in bands_src]
                bands = np.vstack(bands_array)

                # get different field ids in tif
                field_ids = list(np.unique(field_data))
                field_ids.remove(0)

                for fid in field_ids:

                    # get pixels with crop label per band
                    crop_bands = bands[:, field_data == fid].T

                    # add crop index and field id (for csv dataframe stuffs)
                    crop_bands = np.hstack([
                        [[fid] for _ in range(crop_bands.shape[0])], 
                        crop_bands, 
                    ])

                    data.extend(crop_bands.tolist())

            # print(np.array(data).shape) print shape to ensure total number of pixel is correct

            return data


        preprocessed_data_df = pd.DataFrame(columns = ["field_id", *selected_bands])
        #{
        #    "bands": selected_bands,
        #    "crops": defaultdict(lambda: {})
        #}

        folder_ids = get_folder_ids(data_dir, dataset_name, test_label_collection)
        
        # preprocessed_data["crops"][crops[crop]]["data"] = get_all_band_pixel_for_crop(folder_ids, int(crop))
        preprocessed_data_df = pd.concat([
            preprocessed_data_df,
            pd.DataFrame(
                get_all_band_pixel_for_field_id(folder_ids), 
                columns = ["field_id", *selected_bands])
        ], ignore_index=True)


        save_file = f"{get_dir(data_dir, 'preprocessed')}/tabular_test.csv"
        preprocessed_data_df.to_csv(save_file, index=False)
        logging.info(f"Saved preprocessed files to {save_file}")
        # with open(save_file, 'w') as f:
        #    json.dump(preprocessed_data, f)
        #    logging.info(f"Saved preprocessed files to {save_file}")

    else:
        raise Exception("model not supported. Try models in the list ['tabular']")



if __name__ == "__main__":
    fire.Fire(main)
