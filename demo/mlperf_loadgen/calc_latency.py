#!/usr/bin/env python3

import sys
import math


def main(file_path):
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            values.append(int(line))

    values.sort()
    print(values)

    print("===========")
    print(f"50  : {values[len(values) // 2]}")
    print(f"90  : {values[len(values) * 9 // 10]}")
    print(f"99  : {values[len(values) * 99 // 100]}")
    print(f"max : {values[len(values) - 1]}")


def calc_latency_erlang(qps, handling_time, num_workers):
    p_val = [None] * 1000
    q = qps * handling_time
    assert q / num_workers < 1  # Criterion of finite queue

    ##
    ## calc P[0] = ( 1 + q^1/1! + q^2/2! + ... + q^n/n! + q^(n+1)/n!/(n - q) ) ^ -1
    ##      P[k] = q^k/k! * P[0]           for k in [1, n]
    ##    P[n+k] = q^(n+k)/n!/n^k * P[0]   for k in [1, inf]
    ##
    ## recuкrent formula
    ##      P[0] = ( 1 + q^1/1! + q^2/2! + ... + q^n/n! + q^(n+1)/n!/(n - q) ) ^ -1
    ##      P[k] = q * P[k-1] / k          for k in [1, n]
    ##    P[n+k] = q * P[k-1] / n          for k in [n, inf]
    #

    p_val[0] = 1
    cur_ = 1
    for i in range(1, num_workers + 1):
        cur_ *= q / i
        p_val[0] += cur_
    cur_ *= q / (num_workers - q)
    p_val[0] += cur_
    p_val[0] = 1 / p_val[0]

    for i in range(1, num_workers + 1):
        p_val[i] = p_val[i - 1] * q / i

    for i in range(num_workers + 1, 1000):
        p_val[i] = p_val[i - 1] * q / num_workers

    acc = 0
    i = 0
    while acc < 0.99 and i < 1000:
        acc += p_val[i]
        i += 1

    print(f"Order in queue (p99) : {i}")
    print("="*10)

    if i < num_workers:
        latency = handling_time
    else:
        latency = ((2 * (i - num_workers) + 1) / num_workers / 2 + 1) * handling_time

    print(f"Latency (p99) : {latency * 1000} ms")
    print("="*10)


    # Create status
    # sum p[n+1]...p[n+k] while it less pp
    #   [0, n-1]  queue is empty, wait time 0
    #   [n]       I'm first in queue, waiting for first idle worker. Will wait
    #             for handling_time/n/2.
    #   [n+1]     I'm second in queue, waiting for second idle worker. Will wait
    #             for handling_time * (1/n + 1/n/2)
    #   [n+k]     I'm second in queue, waiting for second idle worker. Will wait
    #             for handling_time * (2k + 1) / n / 2
    #
    # def wait_time_percentile(pp):
    #     state = 0

    # Intel demonstrate, is it possible?
    #   2s x 40c
    #
    #   Offline:  4038.5
    #   Server :  3051.10
    #
    #   ================================================
    #   Additional Stats
    #   ================================================
    #   Completed samples per second    : 3051.08
    #
    #   Min latency (ns)                : 5109340
    #   Max latency (ns)                : 31508403
    #   Mean latency (ns)               : 6704363
    #   50.00 percentile latency (ns)   : 6016629
    #   90.00 percentile latency (ns)   : 8715637
    #   95.00 percentile latency (ns)   : 9938075
    #   97.00 percentile latency (ns)   : 10867504
    #   99.00 percentile latency (ns)   : 13019763
    #   99.90 percentile latency (ns)   : 20069872
    #
    #

    # Create status
    # sum p[n+1]...p[n+k] while it less pp
    #   [0, n-1]  queue is empty, Processing time : handling_time
    #   [n]       I'm first in queue, waiting for first idle worker. Will wait
    #             for n*handling_time/2. Processing time : handling_time * (1 + 1/n)
    #   [n+1]     I'm second in queue, waiting for second idle worker. Will wait
    #             (k/n + 1) * handling_time  = result
    # def get_percentile(pp):

    # p[0] =


def calc_latency_simulation(qps, handling_time, num_workers):
    num_of_samples = 10000

    for _ in range(num_of_samples):

        curr_ts =



    # Intel demonstrate, is it possible?
    #   2s x 40c
    #
    #   Offline:  4038.5
    #   Server :  3051.10
    #
    #   ================================================
    #   Additional Stats
    #   ================================================
    #   Completed samples per second    : 3051.08
    #
    #   Min latency (ns)                : 5109340
    #   Max latency (ns)                : 31508403
    #   Mean latency (ns)               : 6704363
    #   50.00 percentile latency (ns)   : 6016629
    #   90.00 percentile latency (ns)   : 8715637
    #   95.00 percentile latency (ns)   : 9938075
    #   97.00 percentile latency (ns)   : 10867504
    #   99.00 percentile latency (ns)   : 13019763
    #   99.90 percentile latency (ns)   : 20069872
    #
    #
if __name__ == '__main__':
    # main(sys.argv[1])
    qps = 1500
    handling_time = 0.005
    num_workers = 10

    calc_latency_erlang(qps=qps, handling_time=handling_time, num_workers=num_workers)
    calc_latency_simulation(qps=qps, handling_time=handling_time, num_workers=num_workers)
