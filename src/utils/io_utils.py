from pathlib import Path
import json

def ensure_dir(path:str)->None:
    Path(path).mkdir(parents=True, exist_ok=True)

def write_json(path:str,obj)->None:
    ensure_dir(str(Path(path).parent))
    with open(path,'w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2)
