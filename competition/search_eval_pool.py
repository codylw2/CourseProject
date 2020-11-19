import math
import os
import sys
import time
import json

import metapy
import pytoml

from multiprocessing import Pool
from functools import partial

script_dir = os.path.dirname(__file__)


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
    else:
        raise Exception('Unknown ranker')


def rank_results(ranker, query_file, idx, doc_dict):
    top_k = 1000
    with open(os.path.join(script_dir, '..', 'predictions.txt'), 'w') as txt:
        query = metapy.index.Document()
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)

            for res in results:
                doc_id = doc_dict['id'][str(res[0])]
                score = res[1]
                txt.write('{} {} {}\n'.format(query_num+1, doc_id, score))

    return


if __name__ == '__main__':

    t_start = time.time()

    cfg = 'config.toml'

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)
    query_cfg = cfg_d['query-runner']

    query_path = query_cfg.get('query-path', 'queries.txt')
    with open(query_path, 'r') as fp:
        query_file = fp.readlines()

    with open('docs.json', 'r') as json_f:
        doc_dict = json.load(json_f)

    print('making inverted index...')
    idx = metapy.index.make_inverted_index(cfg)

    ranker_str = 'bm25'

    if ranker_str == 'dp':
        params = [91.82]
    elif ranker_str == 'jm':
        params = [0.38]
    elif ranker_str == 'ad':
        params = [1.31]
    elif ranker_str == 'bm25':
        params = [2.3, .84, 0]
    else:
        raise Exception('Unknown ranker')

    print('loading ranker...')
    ranker = load_ranker(cfg, ranker_str, params)

    print('ranking docs...')
    rank_results(ranker, query_file, idx, doc_dict)
