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
        return sd.query_term_weight*idf*t_sc


def load_params(ranker_str):
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
        for k in range(1, 11):
            for l in range(1, 11):
                params.append([1.0, 0.38, 1.0, k*.1, l*.1])
        prev_ndcg = 0
        best_params = [0]*5
    elif ranker_str == 'dp':
        params = list()
        for i in range(0, 2001):
            params.append([i*.01])
        prev_ndcg = 0
        best_params = [0]
    elif ranker_str == 'kldprf_dp':
        params = list()
        for i in range(0, 201):
            for j in range(0, 11):
                for k in range(0, 11):
                    params.append([i*.01, j*.1, k*.1])
        prev_ndcg = 0
        best_params = [0, 0, 0]
    elif ranker_str == 'kldprf_jm':
        params = list()
        for i in range(0, 201):
            for j in range(0, 11):
                for k in range(0, 11):
                    params.append([i*.01, j*.1, k*.1])
        prev_ndcg = 0
        best_params = [0, 0, 0]
    else:
        raise Exception('Unknown ranker')
    return params, prev_ndcg, best_params


def load_weights(params):
    param_list = list()
    for i in range(0, 21):
        for j in range(0, 21):
            for k in range(0, 11):
                param_list.append(params + [i*.1, j*.1, k*.1])
    return param_list


def load_ranker(cfg_file, ranker_str, params, fwd_idx):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0)
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    if ranker_str == 'bm25':
        return metapy.index.OkapiBM25(k1=params[0], b=params[1], k3=params[2])
    elif ranker_str == 'bm25p':
        return BM25p(*params)
    elif ranker_str == 'dp':
        return metapy.index.DirichletPrior(params[0])
    elif ranker_str == 'jm':
        return metapy.index.JelinekMercer(params[0])
    elif ranker_str == 'ad':
        return metapy.index.AbsoluteDiscount(params[0])
    elif ranker_str == 'rocchio':
        kwargs = {
            'fwd': fwd_idx,
            'initial_ranker': metapy.index.OkapiBM25(k1=params[0], b=params[1], k3=params[2]),
            'alpha': 1.0,
            'beta': .8,
            'k': 100,
            'max_terms': 10
        }
        return metapy.index.Rocchio(**kwargs)
    elif ranker_str == 'kldprf_dp':
        kwargs = {
            'fwd': fwd_idx,
            'lm_ranker': metapy.index.DirichletPrior(params[0]),
            'alpha': .5,
            'lambda': .5,
            'k': 10,
            'max_terms': 50
        }
        return metapy.index.KLDivergencePRF(**kwargs)
    elif ranker_str == 'kldprf_jm':
        kwargs = {
            'fwd': fwd_idx,
            'lm_ranker': metapy.index.JelinekMercer(params[0]),
            'alpha': .5,
            'lambda': .5,
            'k': 10,
            'max_terms': 50
        }
        return metapy.index.KLDivergencePRF(**kwargs)
    else:
        raise Exception('Unknown ranker')


def rank_results(ranker, query_file, idx, doc_list, results_dict, data_key):
    top_k = 1000
    query = metapy.index.Document()
    for query_num, line in enumerate(query_file):
        query.content(line.strip())
        results = ranker.score(idx, query, top_k)

        if query_num not in results_dict.keys():
            results_dict[query_num] = dict()

        for doc_idx, score in results:
            if doc_idx not in results_dict[query_num].keys():
                results_dict[query_num][doc_idx] = dict()
            results_dict[query_num][doc_idx][data_key] = score

    return results_dict


def calc_ndcg(cfg, results_dict, data_keys, weights):
    d_weights = list(zip(data_keys, weights))

    top_k = 20
    ev = metapy.index.IREval(cfg)

    ndcg = 0.0
    num_queries = 0
    for q_idx, results in results_dict.items():
        results_vector = list()
        for uid, scores in results.items():
            score = sum([scores.get(d_key, 0) * weight for d_key, weight in d_weights])
            results_vector.append([score, uid])

        results_vector = [tuple(r[::-1]) for r in sorted(results_vector, reverse=True)[:top_k]]

        ndcg += ev.ndcg(results_vector, q_idx+1, top_k)
        num_queries += 1
    ndcg = ndcg / num_queries
    return weights, ndcg


if __name__ == '__main__':

    t_start = time.time()

    cfg_template = 'config.toml'
    with open(cfg_template, 'r') as fin:
        cfg_template_str = fin.read()

    run_type = 'train'

    ranking_results = dict()
    data_keys = ['title', 'abstract', 'text']
    doc_weights = [.6, .3, .1]
    # data_keys = ['title']
    # doc_weights = [.6]
    for d_key in data_keys:
        format_dict = {'run_type': run_type, 'data_key': d_key}
        print("{run_type} : {data_key}".format(**format_dict))
        cfg_dir = 'cranfield-{run_type}-{data_key}'.format(**format_dict)
        cfg = 'config-{run_type}-{data_key}.toml'.format(**format_dict)
        with open(cfg, 'w') as f_cfg:
            cfg_str = cfg_template_str.format(**format_dict)
            f_cfg.write(cfg_str)
        cfg_d = pytoml.loads(cfg_str)

        query_cfg = cfg_d['query-runner']
        query_path = query_cfg['query-path']
        with open(query_path, 'r') as fp:
            query_file = fp.readlines()

        with open(os.path.join(cfg_dir, cfg_d['uid-order']), 'r') as json_f:
            doc_list = json.load(json_f)['uid_order']

        # print('removing old idx...')
        # if os.path.exists(cfg_d['index']):
        #     shutil.rmtree(cfg_d['index'])

        print('making inverted index...')
        idx = metapy.index.make_inverted_index(cfg)
        fwd_idx = metapy.index.make_forward_index(cfg)

        ranker_str = 'bm25p'
        params = [1.0, 0.38, 1.0]

        print('loading ranker...')
        ranker = load_ranker(cfg, ranker_str, params, fwd_idx)

        # print('removing old config...')
        # if os.path.exists(cfg):
        #     os.remove(cfg)

        print('ranking docs...')
        ranking_results = rank_results(ranker, query_file, idx, doc_list, ranking_results, d_key)

        print()

    weights = load_weights([])

    print('launching processes...')
    prev_ndcg = 0
    best_params = [0, 0, 0]
    # procs = list()
    # with Pool(8) as p:
    #     for weight_set in weights:
    #         procs.append(p.apply_async(calc_ndcg, (cfg, ranking_results, data_keys, weight_set)))
    #
    #     for proc in procs:
    #         params, ncdg = proc.get()
    #         if prev_ndcg < ncdg:
    #             prev_ndcg = ncdg
    #             best_params = params
    #             print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))

    for weight_set in weights:
        params, ncdg = calc_ndcg(cfg, ranking_results, data_keys, weight_set)
        if prev_ndcg < ncdg:
            prev_ndcg = ncdg
            best_params = params
            print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))

    print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))

    print('script ran in {} seconds'.format(time.time() - t_start))












    #
    # t_start = time.time()
    #
    # os.chdir('cranfield_train')
    #
    # cfg = 'config.toml'
    #
    # with open(cfg, 'r') as fin:
    #     cfg_d = pytoml.load(fin)
    # query_cfg = cfg_d['query-runner']
    #
    # query_path = query_cfg.get('query-path', 'queries.txt')
    # with open(query_path, 'r') as fp:
    #     query_file = fp.readlines()
    #
    # with open(os.path.join('cranfield', 'cranfield-dat.json'), 'r') as json_f:
    #     doc_list = json.load(json_f)['uid_order']
    #
    # print('removing old idx...')
    # if os.path.exists(os.path.join('idx')):
    #     shutil.rmtree(os.path.join('idx'))
    #
    # print('making inverted index...')
    # idx = metapy.index.make_inverted_index(cfg)
    # fwd_idx = metapy.index.make_forward_index(cfg)
    #
    # ranker_str = 'kldprf_jm'
    # if ranker_str == 'bm25':
    #     params = list()
    #     for i in range(0, 20):
    #         for j in range(1, 101):
    #             params.append([i*.1, j*.01, 0])
    #     prev_ndcg = 0
    #     best_params = [0, 0, 0]
    # elif ranker_str == 'bm25p':
    #     params = list()
    #     for i in range(0, 20):
    #         for j in range(1, 101):
    #             for k in range(0, 11):
    #                 params.append([i*.1, j*.01, k*.1])
    #     prev_ndcg = 0
    #     best_params = [0, 0, 0]
    # elif ranker_str == 'rocchio':
    #     params = list()
    #     # for i in range(1, 30):
    #     #     for j in range(1, 101):
    #     #         for k in range(1, 11):
    #     #             for l in range(1, 11):
    #     #                 params.append([i*.1, j*.01, 0, k*.1, l*.1])
    #     for k in range(1, 11):
    #         for l in range(1, 11):
    #             params.append([1.0, 0.38, 1.0, k*.1, l*.1])
    #     prev_ndcg = 0
    #     best_params = [0]*5
    # elif ranker_str == 'dp':
    #     params = list()
    #     for i in range(0, 2001):
    #         params.append([i*.01])
    #     prev_ndcg = 0
    #     best_params = [0]
    # elif ranker_str == 'kldprf_dp':
    #     params = list()
    #     for i in range(0, 201):
    #         for j in range(0, 11):
    #             for k in range(0, 11):
    #                 params.append([i*.01, j*.1, k*.1])
    #     prev_ndcg = 0
    #     best_params = [0, 0, 0]
    # elif ranker_str == 'kldprf_jm':
    #     params = list()
    #     for i in range(0, 201):
    #         for j in range(0, 11):
    #             for k in range(0, 11):
    #                 params.append([i*.01, j*.1, k*.1])
    #     prev_ndcg = 0
    #     best_params = [0, 0, 0]
    #
    # print('launching processes...')
    # procs = list()
    # with Pool() as p:
    #     for param in params:
    #         procs.append(p.apply_async(calc_ndcg, (param, cfg, query_cfg, query_file, ranker_str)))
    #
    #     for proc in procs:
    #         params, ncdg = proc.get()
    #         if prev_ndcg < ncdg:
    #             prev_ndcg = ncdg
    #             best_params = params
    #             print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))
    #
    # print('ranker: {2} : PARAM {0} : NDCG {1}'.format(best_params, prev_ndcg, ranker_str))
    # print('script ran in {} seconds'.format(time.time()-t_start))













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



