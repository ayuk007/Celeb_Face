import os
import sys
import shutil
from tqdm import tqdm
from bing_image_downloader import downloader

celebs = os.listdir("Dataset/")
celebs_new = os.listdir("Datasets_New/")
celebs = list(set(celebs) ^ set(celebs_new))

for celeb in tqdm(celebs):
    downloader.download(celeb, output_dir="Datasets_New", limit = 70,
                        timeout = 3, filter = "photo", adult_filter_off=False, verbose = False)
    
celeb_names = os.listdir("Datasets_New/")

for celeb in tqdm(celeb_names):
    files = os.listdir(f"Datasets_New/{celeb}")
    for ind in range(len(files)+1):
        os.rename(os.path.join(f"Datasets_New/{celeb}", files[ind]), f"Datasets_New/{celeb}/{files[ind]}_{ind}")

print("Finished!!")