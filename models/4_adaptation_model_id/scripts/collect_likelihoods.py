import os
import glob
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir_root", required=True)
    
    args = parser.parse_args()
    
    model_paths = os.path.join(args.out_dir_root, "") + "subject*/*/likelihood"
    for p in glob.glob(model_paths):
        with open(p, "r") as l_f:
            likelihood_str = l_f.readlines()[0].strip()
            print(likelihood_str)

if __name__ == '__main__':
    main()
