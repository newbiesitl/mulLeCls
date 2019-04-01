from config import data_folder
import os
import pandas as pd

def jigsaw_toxix_ds_get_df():
    dataset_name = 'jigsaw-toxic-comment-classification-challenge'
    dataset_path = os.path.join(data_folder, dataset_name)
    train_file = os.path.join(dataset_path, 'train.csv')
    train_df = pd.read_csv(train_file, sep=',')
    return train_df