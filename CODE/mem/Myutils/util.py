# -*- coding:UTF-8 -*-
import os
import pickle
import json
from datetime import datetime
import dateutil
import shutil


def save(preproc, dirname):
    print('--------------------saving preproc------------------------')
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'wb') as fid:
        pickle.dump(preproc, fid)

def check_and_make(target_root: str):
    if not os.path.exists(target_root):
        os.makedirs(target_root)

def get_files_path(file_dir: str, file_type: str) -> list:
    """get the file list of a certain type of files, like .mat .npy .png .jpg .xlsx

    Args:
        file_dir (str): the saving dir of the files, note that the dir can contain different types of files
        file_type (str): choose a certain type of files

    Returns:
        list: return the full path of a certain type files
    """
    files = []
    if os.path.exists(file_dir):
        files_list = os.listdir(file_dir)
    else:
        raise IOError(f'the path {file_dir} is invalid')

    for f in files_list:
        if f.split('.')[-1] == file_type:
            files.append(os.path.join(file_dir,f))

    return files

def get_files_name(file_dir: str, file_type: str) -> list:
    """get the list of file names of a certain type of files

    Args:
        file_dir (str): the saving dir of the files
        file_type (str): choose a certain type of files

    Returns:
        list: return the names of a certain type of files
    """ 
    names = []
    if os.path.exists(file_dir):
        files_list = os.listdir(file_dir)
    else:
        raise IOError(f'the path {file_dir} is invalid')

    for f in files_list:
        if f.split('.')[-1] == file_type:
            appendix = len(file_type) + 1 # 1 means the decimal
            names.append(f[0:-appendix])
            
    return names

def update_json(json_path: str, content: dict) -> None:
    """function used to update the parameters in config json 

    Args:
        json_path (str): the root of the config json
        content (dict): the update content
    """ 

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                file = json.load(f)
        except:
            raise TypeError(f'the json file {json_path} is invalid')
    
    else:
        raise IOError(f'the json path {json_path} is missing')
    
    for k in content.keys():
        assert k in file.keys(), f'the json file:{json_path} does not contain {k}'
        file[k] = content[k]
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(file, f, indent=2)

def check_path_exists(path: str) -> None:

    if os.path.exists(path):
        pass
    else:
        raise IOError(f'the path {path} is invalid')
    

def create_save_dir(save_dir, project_name, exp_name, clear_all_model_save=False,**params):

    save_dir = os.path.join(save_dir, project_name)
    os.makedirs(save_dir, exist_ok=True)

    if clear_all_model_save:
        print("############# WARNING #############")
        print(f'Clear all model of {project_name} reposited')
        for f in os.listdir(save_dir):
            model_path = os.path.join(save_dir, f)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)

    path_dict = {}
    os.makedirs(save_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(save_dir, exp_name)
    # 返回正确时区的时间
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H:%M')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix, exist_ok=True)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path, exist_ok=True)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path, exist_ok=True)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path, exist_ok=True)
    path_dict['sample_path'] = sample_path

    # set tensorboard envents save path
    events_path = os.path.join(prefix, 'runs')
    os.makedirs(events_path, exist_ok=True)
    path_dict['events_path'] = events_path

    return path_dict