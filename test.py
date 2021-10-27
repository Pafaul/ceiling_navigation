import multiprocessing
import time


def f(x):
    return x**2


def main():
    start_time = time.time()
    pools = multiprocessing.Pool(4)
    result = pools.map(f, list(range(1000000)))
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
