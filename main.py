import os
import shutil
import pandas as pd

source_base_path = "data/random_split"
destination_base_path = "data/random_split"

def extract_csv_files():
    for split in ["train", "dev", "test"]:
        src_folder = os.path.join(source_base_path, split)
        dst_folder = os.path.join(destination_base_path, split)
        os.makedirs(dst_folder, exist_ok=True)
        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, filename)
            shutil.copy(src_file, dst_file)

def load_split(split_name):
    folder_path = os.path.join(destination_base_path, split_name)
    dfs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)[["sequence", "family_accession"]]
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

extract_csv_files()
train_df = load_split("train")
dev_df = load_split("dev")
test_df = load_split("test")

train_df.to_csv("data/train.csv", index=False)
dev_df.to_csv("data/dev.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
