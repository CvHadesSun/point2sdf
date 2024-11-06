
import argparse
import sys
import  os
import time
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sample import compute_sdf_and_occ_points_new,compute_sdf_and_occ_points_new_cpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--count', type=int, default=2500000)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()

    # counts=[10_000,1000_000,10_000_000]
    counts=[100_000_000]


    for c in counts:
        t0 = time.time()
        for i in range(10):
            print('------')
            if args.cuda:
                compute_sdf_and_occ_points_new(args.input,args.output,c,epsilon=args.epsilon)
            else:
                compute_sdf_and_occ_points_new_cpu(args.input,args.output,c,epsilon=args.epsilon)
        t1 = time.time()
        print(f" {c} mean time: {(t1-t0)/10:.4f}s")