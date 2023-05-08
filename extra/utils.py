import json
import pandas as pd

from pathlib import Path
from typing import List, Dict, Union
def load_json(fp: Union[str, Path]):
    import json
    with open(fp, 'r') as f:
        return json.load(f)

def to_dataframe(data: List[Dict]):
    # data: List of dicts
    # to data frame
    df = pd.DataFrame(data)
    return df