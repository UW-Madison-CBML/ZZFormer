import os
import gc
import argparse
import random
import numpy as np
import torch
import pickle
from functools import partial
from torch.utils.data import DataLoader
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# ====================================================================================
# MAIN
# ====================================================================================
def main(args):
    k_mers=[]
    if args.mer2_dir:
        k_mers.append(2)
    if args.mer4_dir:
        k_mers.append(4)
    if args.mer8_dir:
        k_mers.append(8)

    all_pkl = {}

    # MnTEdb
    image_files = glob.glob(f"./{mntedb_path}/*") #CHANGE PATH
    print(image_files)
    for file in image_files:
        with tarfile.open(file, "r:gz") as tar:
            all_files = tar.getnames()
            pkl_path = next((f for f in all_files if f.endswith('.pkl')), None)
            if pkl_path:
                member = tar.getmember(pkl_path)
                f = tar.extractfile(member)
                data = pickle.load(f)
                all_pkl = all_pkl | data







    # ---------------- Resumption & Saving Setup ----------------
    save_dir = args.save_dir 
    os.makedirs(save_dir, exist_ok=True)
    



    # Save file now includes the fold number
    save_path = os.path.join(save_dir, f"{args.mode}_fold{args.fold}_best_{args.run_name}_transformeronly.pt")
    metrics_save_path = os.path.join(save_dir, f"{args.mode}_allfold_metrics.txt")










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mer2_dir', type=str, required=False, help='Path to 2mer topological embeddings')
    parser.add_argument('--mer4_dir', type=str, required=False, help='Path to 4mer topological embeddings')
    parser.add_argument('--mer8_dir', type=str, required=False, help='Path to 8mer topological embeddings')

    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save model checkpoints')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)