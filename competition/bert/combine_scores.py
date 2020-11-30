import argparse
import json
import math
import numpy as np
import os
import time

from functools import reduce


def write_json(json_dict, f_name, output_dir, indent=False):
    print('writing json: {0}'.format(os.path.join(output_dir, f_name)))
    with open(os.path.join(output_dir, f_name), 'w') as json_f:
        if indent:
            json.dump(json_dict, json_f, indent=4)
        else:
            json.dump(json_dict, json_f)


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def gather_files(score_dir):
    print('gathering files: {}'.format(score_dir))
    f_list = list()
    for f in os.listdir(score_dir):
        f_path = os.path.join(score_dir, f)
        if os.path.isfile(f_path) and os.path.splitext(f_path)[1] == '.json':
            f_list.append(f_path)
    return f_list


def combine_results(f_list):
    print('combining results...')
    min_score = 0
    max_score = 0

    ranking_dict = dict()
    for f in f_list:
        with open(f, 'r') as f_json:
            tmp_dict = json.load(f_json)
        for q_idx, doc_list in tmp_dict.items():
            q_int = int(q_idx)
            if q_int not in ranking_dict.keys():
                ranking_dict[q_int] = dict()

            for doc_dict in doc_list:
                d_uid = doc_dict['d_uid']
                if d_uid not in ranking_dict[q_int].keys():
                    ranking_dict[q_int][d_uid] = {'chunks': list()}
                ranking_dict[q_int][d_uid]['chunks'].append([doc_dict['chunk'], doc_dict['score']])

                if doc_dict['score'] < min_score:
                    min_score = doc_dict['score']
                if doc_dict['score'] > max_score:
                    max_score = doc_dict['score']

    print('max score: {}'.format(max_score))
    print('min score: {}'.format(min_score))

    return ranking_dict, min_score


def score_docs(ranking_dict, min_score):
    print('determining combined scoring...')
    score_pad = 1+abs(min_score)
    for q_idx, all_docs in ranking_dict.items():
        for d_uid, doc_dict in all_docs.items():
            scores = [s[1] for s in doc_dict['chunks']]
            doc_dict['sum'] = math.fsum(scores)
            doc_dict['mean'] = doc_dict['sum'] / len(scores)
            doc_dict['max'] = max(scores)

            g_scores = [g+score_pad for g in scores]
            doc_dict['gmean'] = geo_mean(g_scores)

            all_docs[d_uid] = doc_dict
        ranking_dict[q_idx] = all_docs
    return ranking_dict


def create_predictions(predict_dir, ranking_dict, pred_key):
    print('creating predictions...')
    for q_idx, all_docs in ranking_dict.items():
        tmp_list = [{'d_uid': d_uid, 'score': doc_dict[pred_key]} for d_uid, doc_dict in all_docs.items()]
        tmp_list.sort(key=lambda x: x['score'], reverse=True)
        ranking_dict[q_idx] = tmp_list[:1000]

    with open(os.path.join(predict_dir, 'predictions.txt'), 'w') as f_predict:
        for key, val in ranking_dict.items():
            for doc in val:
                f_predict.write('{} {} {}\n'.format(key, doc['d_uid'], doc['score']))

    return ranking_dict


# Converts a Time to a human-readable format
# from https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution
def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
           reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                  [(t * 1000,), 1000, 60, 60])


if __name__ == '__main__':
    # Get start time of execution
    startTime = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_input_dir", type=str, help="")
    parser.add_argument("--scores_input_file", type=str, help="")
    parser.add_argument("--scores_output_file", type=str, default='test_scores_comb.json', help="")
    parser.add_argument("--pred_key", type=str, default='max', help="")
    parser.add_argument("--reload", action="store_true", help="")

    args = parser.parse_args()
    print(" * Running with arguments: " + str(args))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    score_dir = os.path.join(script_dir, 'scores')
    if not os.path.isdir(score_dir):
        os.makedirs(score_dir)

    if args.scores_input_dir and args.scores_input_file:
        raise Exception('only specify one of --score_input_dir and --score_input_file')
    if args.reload and not (args.scores_input_dir or args.scores_input_file):
        raise Exception('must specify one of --score_input_dir and --score_input_file')

    if args.reload:
        if args.scores_input_dir:
            file_list = gather_files(args.scores_input_dir)
        else:
            file_list = [args.scores_input_file]
        ranking_results = score_docs(*combine_results(file_list))
        write_json(ranking_results, args.scores_output_file, score_dir)
    else:
        with open(os.path.join(score_dir, args.scores_output_file), 'r') as json_f:
            ranking_results = json.load(json_f)

    create_predictions(score_dir, ranking_results, args.pred_key)

    # Display total execution time
    print(" * Total execution time: " + secondsToStr(time.time() - startTime))
