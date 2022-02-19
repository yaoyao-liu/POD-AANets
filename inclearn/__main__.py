import matplotlib
import os
from inclearn import parser
from inclearn.train import train
from inclearn.gpu_tools import occupy_memory, set_gpu

matplotlib.use('Agg')

def main():
    args = parser.get_parser().parse_args()
    args = vars(args) 
    set_gpu(str(args["device"][0]))
    occupy_memory(args["device"][0])
    if args["seed_range"] is not None:
        args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
        print("Seed range", args["seed"])

    for _ in train(args):
        pass
        
if __name__ == "__main__":
    main()
