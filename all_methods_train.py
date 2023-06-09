import sys
import os


if __name__ == '__main__':
    methods = [
        "gd",
        "opt_gd",
        "manual_gd",
        "em",
        "emgd",
        "lmoments",
        "moments",
    ]

    for method in methods:
        os.system(f"python train.py {method}")