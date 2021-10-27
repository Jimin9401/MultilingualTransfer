import dataclasses
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%Sgp',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CFExample(object):

    guid: str
    input_ids: List[int]
    category_id: int

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


@dataclass
class NMTExample(object):
    guid: str
    input_ids: List[int]
    trg_ids: List[int]

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"