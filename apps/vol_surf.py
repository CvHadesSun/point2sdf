
import argparse
import sys
import  os
import time
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from src.sample import compute_sdf_and_occ_points_new,compute_sdf_and_occ_points_new_cpu
from src.sample import sample_volume_and_surface
import warnings
warnings.filterwarnings("ignore", message="No mtl file provided")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--vol_count', type=int, default=2500000)
    parser.add_argument('--surf_count', type=int, default=2500000)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()


    if args.cuda:
        sample_volume_and_surface(args.input,args.output,args.vol_count,args.surf_count,args.epsilon)
    else:
        raise NotImplementedError("CPU version not implemented yet")