import os
from os.path import isfile, join, split


from torch.hub import download_url_to_file
import zipfile

c_dir = split(__file__)[0]

def download_raft_models(models_base_path=None):
    if models_base_path is None:
        models_base_path = c_dir
    models_path = join(models_base_path, "models/")
    
    if not isfile( join(models_path,"raft-sintel.pth")):
        print(f"RAFT pre-trained models not present!")
        print(f"  Starting download and extrection to {models_path}")
    
    
        
        os.makedirs(models_path, exist_ok=True)
        zip_path = join(models_path, "models.zip")
        download_url_to_file('https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip?dl=1',zip_path)

        with zipfile.ZipFile(zip_path) as fp:
            fp.extractall(join(models_base_path))
    else:
        print("RAFT pre-trained models already present")    

    # md5sum   models/raft-sintel.pth     cc69e5da1f38673ab10d1849859ebe91
