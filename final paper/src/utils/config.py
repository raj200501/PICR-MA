from pathlib import Path
import yaml

def load_yaml(path:str):
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def dump_yaml(obj,path:str):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        yaml.safe_dump(obj,f,sort_keys=False)
