from .vocabulary import *
from .functions import *

import torch

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
