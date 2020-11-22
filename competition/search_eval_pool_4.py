import math
import os
import sys
import time

import metapy
import pytoml

from multiprocessing import Pool
from functools import partial


def load_ranker(cfg_file, ranker_str, params):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    if ranker_str == 'bm25':
        return metapy.index.OkapiBM25(k1=params[0], b=params[1], k3=params[2])
    elif ranker_str == 'dp':
        return metapy.index.DirichletPrior(params[0])
    elif ranker_str == 'jm':
        return metapy.index.JelinekMercer(params[0])
    elif ranker_str == 'ad':
        return metapy.index.AbsoluteDiscount(params[0])


def calc_ndcg(params, cfg, query_cfg, query_file, ranker_str, idx):
    # print(params)

    ranker = load_ranker(cfg, ranker_str, params)
    ev = metapy.index.IREval(cfg)

    top_k = 20
    query_start = query_cfg.get('query-id-start', 0)

    ndcg = 0.0
    num_queries = 0
    query = metapy.index.Document()
    for query_num, line in enumerate(query_file):
        query.content(line.strip())
        results = ranker.score(idx, query, top_k)
        # avg_p = ev.avg_p(results, query_start + query_num, top_k)
        # print("Query {} average precision: {}".format(query_num + 1, avg_p))

        ndcg += ev.ndcg(results, query_start + query_num, top_k)
        num_queries += 1
    ndcg = ndcg / num_queries

    # map = ev.map()
    print("{} : NDCG: {}".format(params, ndcg))
    return params, ndcg


if __name__ == '__main__':

    t_start = time.time()

    cfg = 'config.toml'

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)
    query_cfg = cfg_d['query-runner']

    query_path = query_cfg.get('query-path', 'queries.txt')
    with open(query_path, 'r') as fp:
        query_file = fp.readlines()

    idx = metapy.index.make_inverted_index(cfg)

    # for ranker_str in ['dp', 'jm', 'ad', 'bm25']:
    # for ranker_str in ['dp', 'jm', 'ad']:
    # for ranker_str in ['bm25']:
    for ranker_str in ['jm']:
    # for ranker_str in ['dp']:
    # for ranker_str in ['ad']:
        if ranker_str == 'bm25':
            prev_ndcg = 0
            best_params = [1.2, .75, 100]
            factors = [.1, .01, 5]
        else:
            prev_ndcg = 0
            best_params = [0]

        p_calc_map = partial(calc_ndcg, cfg=cfg, query_cfg=query_cfg, query_file=query_file, ranker_str=ranker_str, idx=idx)

        outfile = open('results_{}_full.txt'.format(ranker_str), 'w')
        outfile.close()
        print('launching processes for: {}'.format(ranker_str))
        if ranker_str != 'bm25':
            for i in range(1, 10001, 1):
                print('{} : {}'.format(ranker_str, i))
                res = calc_ndcg([.01 * i], cfg=cfg, query_cfg=query_cfg, query_file=query_file, ranker_str=ranker_str, idx=idx)
                if prev_ndcg < res[1]:
                    prev_ndcg = res[1]
                    best_params = res[0]

        else:

            params = [0, 0, 0]
            for i in range(1, 16):
                for j in range(1, 101):
                    for k in range(1, 1000, 1):
                        params[0] = factors[0] * i + 1
                        params[1] = factors[1] * j
                        params[2] = factors[2] * k
                        res = calc_ndcg(params, cfg=cfg, query_cfg=query_cfg, query_file=query_file,
                                        ranker_str=ranker_str, idx=idx)
                        if prev_ndcg < res[1]:
                            prev_ndcg = res[1]
                            best_params = res[0]
                    outfile = open('results_{}_full.txt'.format(ranker_str), 'a')
                    outfile.write('ranker: {2} : PARAM {0} : NDCG {1}\n'.format(best_params, prev_ndcg, ranker_str))
                    outfile.close()
                    print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))

        print('final: ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))
        print('elapsed time {0} seconds'.format(time.time() - t_start))
        outfile = open('results_{}_full.txt'.format(ranker_str), 'a')
        outfile.write('final: ranker: {2} : PARAM {0} : NDCG {1}\n'.format(best_params, prev_ndcg, ranker_str))
        outfile.close()


    # BM25
    # i 115 : PARAM [1.1500000000000001, 1.05, 5] : NDCG 0.3539291924803032



