import math
import os
import sys
import time
import json

import metapy
import pytoml
import shutil

script_dir = os.path.dirname(__file__)


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
        return idf*t_sc


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


def rank_results(ranker, query_file, idx, doc_list):
    top_k = 1000
    with open(os.path.join(script_dir, '..', 'predictions.txt'), 'w') as txt:
        query = metapy.index.Document()
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)

            score_dict = dict()
            for res in results:
                doc_id = doc_list[res[0]]
                score = res[1]
                if score not in score_dict.keys():
                    score_dict[score] = []
                score_dict[score].append(doc_id)

            for score in sorted(list(score_dict.keys()), reverse=True):
                for doc_id in sorted(score_dict[score]):
                    txt.write('{} {} {}\n'.format(query_num+1, doc_id, score*4))

    return


if __name__ == '__main__':

    t_start = time.time()

    os.chdir('cranfield_test')

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
    shutil.rmtree(os.path.join('idx'))

    print('making inverted index...')
    idx = metapy.index.make_inverted_index(cfg)
    fwd_idx = metapy.index.make_forward_index(cfg)

    ranker_str = 'bm25p'

    if ranker_str in ['dp', 'kldprf_dp']:
        params = [91.82]
    elif ranker_str in ['jm', 'kldprf_jm']:
        params = [0.38]
    elif ranker_str == 'ad':
        params = [1.31]
    elif ranker_str in ['bm25', 'kldprf_bm25', 'rocchio']:
        # params = [2.3, .84, 0]
        # params = [0.1, 0.95, 0]
        # params = [0.9, 0.52, 0]  # params 3
        # params = [1.2, 0.37, 0]  # params 4
        # params = [1.6, 0.75, 0]  # params 5
        params = [0.25, 0.35, 0]  # params 6
    elif ranker_str == 'bm25p':
        params = [0.6, 0.39, 0.7]  # params 1
    else:
        raise Exception('Unknown ranker')

    print('loading ranker...')
    ranker = load_ranker(cfg, ranker_str, params, fwd_idx)

    print('ranking docs...')
    rank_results(ranker, query_file, idx, doc_list)

    # expected run time 150 seconds
    print('script ran in {} seconds'.format(time.time()-t_start))
