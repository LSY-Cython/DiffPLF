import json
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

def create_pred_dataset(input_folder, output_path):
    pkl_paths = [f for f in os.listdir(input_folder) if f.endswith("pkl")]
    # use three-year data for training
    train_paths = [f"{input_folder}/{f}" for f in pkl_paths if f.split("-")[0] != "2019"]
    test_paths = [f"{input_folder}/{f}" for f in pkl_paths if f.split("-")[0] == "2019"]
    with open(output_path, "w") as f:
        json.dump({"train": train_paths, "test": test_paths}, f, indent=4)

def process_input(path, isConstrain):
    with open(path, "rb") as f:
        input_data = pkl.load(f)
    hist = input_data["history"]
    true = input_data["profile"]
    ev_num = input_data["evs"]
    day_type = input_data["day"]
    temp = input_data["temp"]
    rhum = input_data["rhum"]
    # scaling
    p_max = np.max(hist)
    num_norm = ev_num/200
    hist_norm = torch.tensor(hist/p_max, dtype=torch.float32).transpose(0, 1)
    if isConstrain:
        true_norm = torch.tensor(true/(p_max*num_norm), dtype=torch.float32).view(-1, 1)  # tackle magnitude imbalance issue
    else:
        true_norm = torch.tensor(true/p_max, dtype=torch.float32).view(-1, 1)
    temp_norm = torch.tensor(temp/42, dtype=torch.float32).view(1, -1)
    rhum_norm = torch.tensor(rhum/100, dtype=torch.float32).view(1, -1)
    weather_norm = torch.cat((temp_norm, rhum_norm)).transpose(0, 1)
    num_norm = torch.tensor([num_norm], dtype=torch.float32)
    # one-hot
    day_vec = torch.zeros(7)
    day_vec[day_type-1] = 1
    return true_norm, hist_norm, p_max, weather_norm, day_vec, num_norm

class PredData(Dataset):
    def __init__(self, data_file, isRefine, isConstrain):
        with open(data_file, "r") as f:
            self.data_paths = json.load(f)["train"]
        self.isRefine = isRefine
        self.isConstrain = isConstrain

    def __getitem__(self, index):
        path = self.data_paths[index]
        true_norm, hist_norm, p_max, weather_norm, day_vec, num_norm = process_input(path, self.isConstrain)
        if self.isRefine:
            path = path.replace("daily", "median")
            with open(path, "rb") as f:
                median_data = pkl.load(f)
            median_norm = torch.tensor(median_data/(p_max*(num_norm.item())), dtype=torch.float32)  # (B, L, 1)
            output_data = {"true": true_norm,  # (B, L, 1)
                           "hist": hist_norm,  # (B, L, N)
                           "weather": weather_norm,  # (B, L, 2)
                           "day": day_vec,  # (B, 7)
                           "num": num_norm,  # (B, 1)
                           "median": median_norm  # (B, L, 1)
                           }
        else:
            output_data = {"true": true_norm,  # (B, L, 1)
                           "hist": hist_norm,  # (B, L, N)
                           "weather": weather_norm,  # (B, L, 2)
                           "day": day_vec,  # (B, 7)
                           "num": num_norm  # (B, 1)
                           }
        return output_data

    def __len__(self):
        return len(self.data_paths)

def creat_dataloader(data_file, batch_size, shuffle, isRefine, isConstrain):
    dataset = PredData(data_file, isRefine, isConstrain)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader

if __name__ == "__main__":
    # create_pred_dataset("dataset/daily", "dataset/pred_dataset.json")
    # loader = creat_dataloader("dataset/pred_dataset.json", 8, shuffle=True)
    pass