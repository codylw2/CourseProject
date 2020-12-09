import argparse
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
        return sd.query_term_weight*idf*t_sc


def load_params(ranker_str):
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
    elif ranker_str == 'bm25+':
        # params = [0.6, 0.39, 0.7]  # params 1
        # params = [0.4, 0.36, 0.7]  # params 2
        params = [1.0, 0.38, 1.0]  # params 3
    else:
        raise Exception('Unknown ranker')
    return params


def load_ranker(cfg_file, ranker_str, params, fwd_idx):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    if ranker_str == 'bm25':
        return metapy.index.OkapiBM25(*params)
    elif ranker_str == 'bm25+':
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
            'initial_ranker': metapy.index.OkapiBM25(*params),
            'alpha': 1.0,
            'beta': .8,
            'k': 100,
            'max_terms': 10
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


def load_doc_weights():
    # return [.6, .3, .1]  # weights_1
    return [0.8, 0.5, 0.0]


def rank_results(ranker, query_file, idx, doc_list, results_dict, data_key):
    top_k = 10000
    query = metapy.index.Document()
    for query_num, line in enumerate(query_file):
        query.content(line.strip())
        results = ranker.score(idx, query, top_k)

        if query_num not in results_dict.keys():
            results_dict[query_num] = dict()

        for doc_idx, score in results:
            doc_uid = doc_list[doc_idx]
            if doc_uid not in results_dict[query_num].keys():
                results_dict[query_num][doc_uid] = dict()
            results_dict[query_num][doc_uid][data_key] = score

    return results_dict


def gen_predictions(results_dict, dat_keys, weights, predict_dir):
    d_weights = list(zip(dat_keys.split(';'), weights))
    print(d_weights)

    pred_file = os.path.join(predict_dir, 'predictions.txt')
    with open(pred_file, 'w') as txt:
        for q_idx, results in results_dict.items():
            results_vector = list()
            for uid, scores in results.items():
                score = sum([scores.get(d_key, 0)*weight for d_key, weight in d_weights])
                results_vector.append([score, uid])

            for score, uid in sorted(results_vector, reverse=True)[:5000]:
                txt.write('{} {} {}\n'.format(q_idx + 1, uid, score))


if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Process dataset and produce json of processed data')
    parser.add_argument('--config_template', type=str, default='config.toml', help='/path/to/cranfield/config.toml')
    parser.add_argument('--run_type', type=str, default='train;test', help='the dataset(s) to process, e.g. "train;test"')
    parser.add_argument('--dat_keys', type=str, required=True, help='the dataset(s) to create for processing, e.g. "title;abstract;text"')
    parser.add_argument('--doc_weights', type=str, required=True, help='the weights to apply to each dat_key')
    parser.add_argument('--ranker', type=str, default='bm25+', help='the ranker to use for document ranking')
    parser.add_argument('--params', type=str, default='1.0;0.38;1.0', help='the ranker to use for document ranking')
    parser.add_argument('--cranfield_dir', type=str, default=file_dir,
                        help='the directory that contains the cranfield data')
    parser.add_argument('--predict_dir', type=str, default=os.path.join(file_dir, '..', '..'),
                        help='the directory that contains the cranfield data')
    parser.add_argument('--remove_idx', action='store_true', help='remove and exist inverted index and recreate it')
    args = parser.parse_args()

    t_start = time.time()

    if len(args.dat_keys.split(';')) != len(args.doc_weights.split(';')):
        raise Exception('The number dat keys and doc keys must be equal')

    cfg_template = args.config_template
    with open(cfg_template, 'r') as fin:
        cfg_template_str = fin.read()

    params = [float(p) for p in args.params.split(';')] if args.params else load_params(args.ranker)

    for run_type in str(args.run_type).split(';'):
        ranking_results = dict()

        doc_weights = [float(i) for i in args.doc_weights.split(';')]
        for d_key in args.dat_keys.split(';'):
            format_dict = {'run_type': run_type, 'data_key': d_key}
            print("{run_type} : {data_key}".format(**format_dict))
            cfg_dir = 'cranfield-{run_type}-{data_key}'.format(**format_dict)
            cfg_path = os.path.join(args.cranfield_dir, 'config-{run_type}-{data_key}.toml'.format(**format_dict))
            with open(cfg_path, 'w') as f_cfg:
                cfg_str = cfg_template_str.format(**format_dict)
                f_cfg.write(cfg_str)
            cfg_d = pytoml.loads(cfg_str)

            query_cfg = cfg_d['query-runner']
            query_path = os.path.join(args.cranfield_dir, query_cfg['query-path'])
            with open(query_path, 'r') as fp:
                query_file = fp.readlines()

            with open(os.path.join(args.cranfield_dir, cfg_dir, cfg_d['uid-order']), 'r') as json_f:
                doc_list = json.load(json_f)['uid_order']

            if args.remove_idx:
                print('removing old idx...')
                if os.path.exists(cfg_d['index']):
                    shutil.rmtree(cfg_d['index'])

            print('making inverted index...')
            idx = metapy.index.make_inverted_index(cfg_path)
            fwd_idx = metapy.index.make_forward_index(cfg_path)

            print('loading ranker...')
            ranker = load_ranker(cfg_path, args.ranker, params, fwd_idx)

            print('removing old config...')
            if os.path.exists(cfg_path):
                os.remove(cfg_path)

            print('ranking docs...')
            ranking_results = rank_results(ranker, query_file, idx, doc_list, ranking_results, d_key)

            print()

        gen_predictions(ranking_results, args.dat_keys, doc_weights, args.predict_dir)

    print('script ran in {} seconds'.format(time.time() - t_start))
