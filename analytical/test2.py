import math
import time

import numpy as np
import multiprocessing as mp

print(f"Cores: {mp.cpu_count()}")

a = 100

def worker(x, y):
    return x + y

def main():

    inputs = [{"x": i, "y": 1} for i in range(1, 10)]

    with mp.Pool(mp.cpu_count() - 1) as pool:
        res = pool.starmap(worker, inputs)

    print(type(res))
    print(res)


if __name__ == '__main__':
    main()