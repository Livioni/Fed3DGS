import torch
from scene.gaussian_model import GaussianModel

global_model_path = "outputs/vast_12clients/real_fed_global_models/global_model_epoch15000.pth"
global_params = torch.load(global_model_path)
global_model = GaussianModel(3)
global_model.set_params(global_params)
global_model.save_ply(global_model_path.replace('.pth', '.ply'))