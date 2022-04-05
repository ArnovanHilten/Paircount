import multiprocessing
from multiprocessing import Pool
from time import time

def sum_nums(low, high):
    result = 0
    for i in range(low, high+1):
        result += i
    return result

# map requires a function to handle a single argument
def sn(a):
    return sum_nums(a[0], a[1])

if __name__ == '__main__':
    #t = time()
    # takes forever
    #print sum_nums(1,10**10)
    #print '{} s'.format(time() -t)
    p = Pool(4)

    n = int(1e3)
    r = range(0,10**10+1,n)
    results = []

    # using apply_async
    t = time()
    for arg in zip([x+1 for x in r],r[1:]):
        results.append(p.apply_async(sum_nums, arg))

    # wait for results
    print(sum(res.get() for res in results))
    print('{} s'.format(time() -t))

    # using process pool
    t = time()
    print(sum(p.map(sn, zip([x+1 for x in r], r[1:]))))
    print('{} s'.format(time() -t))