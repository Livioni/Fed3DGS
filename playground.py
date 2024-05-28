from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams  
from gaussian_splatting.scene import Scene, GaussianModel
    
    
if __name__ == "__main__":
    mp = ModelParams()
    mp._source_path = "outputs/10clients/rubble_colmap_results_icp/00000"
    mp._model_path = "outputs/whole_scene/test"
    mp._images = "datasets/rubble-pixsfm/train/rgbs"
    gaussians = GaussianModel(3)
    scene = Scene(mp, gaussians)