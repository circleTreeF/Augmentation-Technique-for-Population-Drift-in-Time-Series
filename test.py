from multiprocessing import Pool
import os, time, random


def long_time_task(name, arg2):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds. %d' % (name, (end - start), arg2))


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    # p = Pool(4)
    with Pool(4) as p:
        arg_list = [10, 20, 30, 40, 50]
        for i in range(5):
            p.apply_async(long_time_task, args=(i, arg_list[i]))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
