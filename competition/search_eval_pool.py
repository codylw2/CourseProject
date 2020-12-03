import math
import json
import os
import sys
import time
import shutil
import metapy
import pytoml

from multiprocessing import Pool
from functools import partial


class BM25p(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, k1=1.2, b=.75, delta=.8):
        self.k1 = k1
        self.b = b
        self.delta = delta
        # You *must* call the base class constructor here!
        super(BM25p, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        idf = math.log((sd.num_docs+1)/sd.doc_count)
        t_sc = (sd.doc_term_count*(self.k1+1))/(sd.doc_term_count+self.k1*(1-self.b+self.b*sd.doc_size/sd.avg_dl))+self.delta
        # return (self.param + sd.doc_term_count) / (self.param * sd.doc_unique_terms + sd.doc_size)
        # tfn = sd.doc_term_count*math.log2(1+sd.avg_dl/sd.doc_size)
        return idf*t_sc


def load_ranker(cfg_file, ranker_str, params, fwd_idx):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0)
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    if ranker_str == 'bm25':
        return metapy.index.OkapiBM25(*params)
    elif ranker_str == 'bm25p':
        return BM25p(*params)
    elif ranker_str == 'dp':
        return metapy.index.DirichletPrior(*params)
    elif ranker_str == 'jm':
        return metapy.index.JelinekMercer(*params)
    elif ranker_str == 'ad':
        return metapy.index.AbsoluteDiscount(*params)
    elif ranker_str == 'rocchio':
        kwargs = {
            'fwd': fwd_idx,
            'initial_ranker': metapy.index.OkapiBM25(*params[:3]),
            'alpha': params[3],
            'beta': params[4],
            'k': 10,
            'max_terms': 50
        }
        return metapy.index.Rocchio(**kwargs)
    elif ranker_str == 'kldprf_dp':
        kwargs = {
            'fwd': fwd_idx,
            'lm_ranker': metapy.index.DirichletPrior(*params),
            'alpha': .5,
            'lambda': .5,
            'k': 10,
            'max_terms': 50
        }
        return metapy.index.KLDivergencePRF(**kwargs)
    elif ranker_str == 'kldprf_jm':
        kwargs = {
            'fwd': fwd_idx,
            'lm_ranker': metapy.index.JelinekMercer(*params),
            'alpha': .5,
            'lambda': .5,
            'k': 10,
            'max_terms': 50
        }
        return metapy.index.KLDivergencePRF(**kwargs)
    else:
        raise Exception('Unknown ranker')


def calc_ndcg(params, cfg, query_cfg, query_file, ranker_str):
    idx = metapy.index.make_inverted_index(cfg)
    fwd_idx = metapy.index.make_forward_index(cfg)

    ranker = load_ranker(cfg, ranker_str, params, fwd_idx)
    ev = metapy.index.IREval(cfg)

    top_k = 20
    query_start = query_cfg.get('query-id-start', 1)

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

    # # map = ev.map()
    # print("{} : NDCG: {}".format(params, ndcg))
    return params, ndcg


if __name__ == '__main__':

    t_start = time.time()

    os.chdir('cranfield_train')

    cfg = 'config.toml'

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)
    query_cfg = cfg_d['query-runner']

    query_path = query_cfg.get('query-path', 'queries.txt')
    with open(query_path, 'r') as fp:
        query_file = fp.readlines()

    with open(os.path.join('cranfield', 'cranfield-dat.json'), 'r') as json_f:
        doc_list = json.load(json_f)['uid_order']

    print('removing old idx...')
    if os.path.exists(os.path.join('idx')):
        shutil.rmtree(os.path.join('idx'))

    print('making inverted index...')
    idx = metapy.index.make_inverted_index(cfg)
    fwd_idx = metapy.index.make_forward_index(cfg)

    ranker_str = 'bm25p'
    if ranker_str == 'bm25':
        params = list()
        for i in range(0, 20):
            for j in range(1, 101):
                params.append([i*.1, j*.01, 0])
        prev_ndcg = 0
        best_params = [0, 0, 0]
    elif ranker_str == 'bm25p':
        params = list()
        for i in range(0, 20):
            for j in range(1, 101):
                for k in range(0, 11):
                    params.append([i*.1, j*.01, k*.1])
        prev_ndcg = 0
        best_params = [0, 0, 0]
    elif ranker_str == 'rocchio':
        params = list()
        # for i in range(1, 30):
        #     for j in range(1, 101):
        #         for k in range(1, 11):
        #             for l in range(1, 11):
        #                 params.append([i*.1, j*.01, 0, k*.1, l*.1])
        for k in range(1, 21):
            for l in range(1, 21):
                params.append([0.1, 0.95, 0, k*.1, l*.1])
        prev_ndcg = 0
        best_params = [0]*5

    print('launching processes...')
    procs = list()
    with Pool() as p:
        for param in params:
            procs.append(p.apply_async(calc_ndcg, (param, cfg, query_cfg, query_file, ranker_str)))

        for proc in procs:
            params, ncdg = proc.get()
            if prev_ndcg < ncdg:
                prev_ndcg = ncdg
                best_params = params
                print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))

    print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))
    print('script ran in {} seconds'.format(time.time()-t_start))




    # for ranker_str in ['dp', 'jm', 'ad', 'bm25']:
    # # for ranker_str in ['dp', 'jm', 'ad']:
    # # for ranker_str in ['bm25']:
    # for ranker_str in ['jm']:
    # # for ranker_str in ['dp']:
    # # for ranker_str in ['ad']:
    #     if ranker_str == 'bm25':
    #         prev_ndcg = 0
    #         best_params = [1.2, .75, 100]
    #         factors = [.1, .01, 5]
    #     else:
    #         prev_ndcg = 0
    #         best_params = [0]
    #
    #     p_calc_map = partial(calc_ndcg, cfg=cfg, query_cfg=query_cfg, query_file=query_file, ranker_str=ranker_str, idx=idx)
    #
    #     outfile = open('results_{}_full.txt'.format(ranker_str), 'w')
    #     outfile.close()
    #     print('launching processes for: {}'.format(ranker_str))
    #     if ranker_str != 'bm25':
    #         for i in range(1, 10001, 1):
    #             print('{} : {}'.format(ranker_str, i))
    #             res = calc_ndcg([.01 * i], cfg=cfg, query_cfg=query_cfg, query_file=query_file, ranker_str=ranker_str, idx=idx)
    #             if prev_ndcg < res[1]:
    #                 prev_ndcg = res[1]
    #                 best_params = res[0]
    #
    #     else:
    #
    #         params = [0, 0, 0]
    #         for i in range(1, 16):
    #             for j in range(1, 101):
    #                 for k in range(1, 1000, 1):
    #                     params[0] = factors[0] * i + 1
    #                     params[1] = factors[1] * j
    #                     params[2] = factors[2] * k
    #                     res = calc_ndcg(params, cfg=cfg, query_cfg=query_cfg, query_file=query_file,
    #                                     ranker_str=ranker_str, idx=idx)
    #                     if prev_ndcg < res[1]:
    #                         prev_ndcg = res[1]
    #                         best_params = res[0]
    #                 outfile = open('results_{}_full.txt'.format(ranker_str), 'a')
    #                 outfile.write('ranker: {2} : PARAM {0} : NDCG {1}\n'.format(best_params, prev_ndcg, ranker_str))
    #                 outfile.close()
    #                 print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))
    #
    #     print('final: ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))
    #     print('elapsed time {0} seconds'.format(time.time() - t_start))
    #     outfile = open('results_{}_full.txt'.format(ranker_str), 'a')
    #     outfile.write('final: ranker: {2} : PARAM {0} : NDCG {1}\n'.format(best_params, prev_ndcg, ranker_str))
    #     outfile.close()


    # BM25
    # i 115 : PARAM [1.1500000000000001, 1.05, 5] : NDCG 0.3539291924803032



