from .model_saver import ModelSaver

import torch

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
