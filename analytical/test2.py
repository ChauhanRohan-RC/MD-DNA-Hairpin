import math
import time

import numpy as np
import multiprocessing as mp

print(f"Cores: {mp.cpu_count()}")

a = 100

def worker(x):

    return np.log(x)

def main():

    inputs = [list(range(i, i + 20)) for i in range(1, 100, 20)]

    pool = mp.Pool(processes=mp.cpu_count())
    res = pool.map(worker, inputs)

    print(type(res))
    print(res)
    np.savetxt("test2.txt", res)


if __name__ == '__main__':
    main()