import argparse
import os
import json
import re
import time
import shutil

from multiprocessing import Pool


def load_json(f_path):
    if os.path.exists(f_path):
        with open(f_path, 'r') as json_f:
            return json.load(json_f)
    else:
        raise Exception('json file does not exist: {0}'.format(f_path))


def load_queries(run_type, input_dir):
    print('loading queries...')
    f_name = '{0}_queries.json'.format(run_type)
    return load_json(os.path.join(input_dir, '..', f_name))


def load_qrels(run_type, input_dir):
    print('loading qrels...')
    f_name = '{0}_qrels.json'.format(run_type)
    return load_json(os.path.join(input_dir, '..', f_name))


def load_docs(run_type, input_dir):
    print('loading docs...')
    f_name = '{0}_docs.json'.format(run_type)
    return load_json(os.path.join(input_dir, '..', f_name))


def load_variants(input_dir):
    f_variants = os.path.join(input_dir, 'corona_variants.txt')
    if os.path.exists(f_variants):
        with open(f_variants, 'r') as f_txt:
            return [line.strip('\n') for line in f_txt.readlines()]


def write_queries(queries, variants, query_keys, run_type):
    print('writing queries...')
    f_query = 'cranfield-{run_type}-queries.txt'.format(run_type=run_type)
    with open(f_query, 'w') as txt:
        for key in sorted([int(k) for k in queries.keys()]):
            query_list = [queries[str(key)][subkey] for subkey in query_keys]
            query_text, _ = update_text(query_list, None, variants)
            txt.writelines(' '.join(query_text) + '\n')


def update_text(text_list, uid, variants):
    re_http = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

    text_list = [re_http.sub('', t.lower()) for t in text_list]

    comp_variants = [re.compile(v) for v in variants]
    for re_var in comp_variants:
        text_list = [re_var.sub('coronavirus', s) for s in text_list]

    # basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=True, split_on_punc=True)
    # tokens = basic_tokenizer.tokenize(text)
    # text = ' '.join(tokens)

    return text_list, uid


def gen_dat(doc_dict, doc_list, variants, doc_keys, run_type, cranfield_dir):
    print('writing .dat files...')
    outfiles = list()
    orderfiles = list()
    for key in doc_keys:
        cranfield_name = 'cranfield-{run_type}-{data_key}'.format(run_type=run_type, data_key=key[0])

        if not os.path.exists(cranfield_name):
            os.makedirs(cranfield_name)
        shutil.copyfile(os.path.join(cranfield_dir, 'line.toml'),
                        os.path.join(cranfield_dir, cranfield_name, 'line.toml'))

        outfiles.append(open(os.path.join(cranfield_dir, cranfield_name, '{0}.dat'.format(cranfield_name)), 'w', encoding='utf-8'))
        orderfiles.append(open(os.path.join(cranfield_dir, cranfield_name, 'cranfield-{run_type}-order.json').format(run_type=run_type), 'w', encoding='utf-8'))

    procs = list()
    with Pool(10) as p:
        for uid in doc_list:
            comb_txt = list()
            for key in doc_keys:
                comb_txt.append(' '.join([doc_dict['uid'][uid][k] for k in key]))

            procs.append(p.apply_async(update_text, (comb_txt, uid, variants)))

        for proc_idx, proc in enumerate(procs):
            processed_text, uid = proc.get()
            for key, text in zip(doc_keys, processed_text):
                if isinstance(key, list):
                    doc_dict['uid'][uid][key[0]] = text
                else:
                    doc_dict['uid'][uid][key] = text

    for uid in doc_list[:]:
        for key, outfile in zip(doc_keys, outfiles):
            outfile.write(doc_dict['uid'][uid][key[0]] + '\n')

    print('number of docs: {0}'.format(len(doc_list)))

    for orderfile in orderfiles:
        json.dump({'uid_order': doc_list}, orderfile)

    [f.close() for f in outfiles]
    [f.close() for f in orderfiles]

    return


def gen_qrels(doc_list, qrels, cranfield_dir):
    print('generating qrels...')
    doc_dict = {uid: idx for idx, uid in enumerate(doc_list)}

    qrels_dst = os.path.join(cranfield_dir, 'cranfield-qrels.txt')

    with open(qrels_dst, 'w') as dst:
        for q_idx, uids in sorted([[int(k), v] for k, v in qrels.items()]):
            for uid, score in uids.items():
                new_line = ' '.join([str(q_idx), str(doc_dict[uid]), str(score)])+'\n'
                dst.write(new_line)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset and produce json of processed data')
    parser.add_argument('--variant_file', type=str, help='/path/to/bert_model/corona_variants.txt')
    parser.add_argument('--variant_default', type=str, default='coronavirus', help='default value to replace corona variants')
    parser.add_argument('--run_type', type=str, default='train;test', help='the dataset(s) to process, e.g. "train;test"')
    parser.add_argument('--query_keys', type=str, required=True, help='the queries to create for processing, e.g. "query"')
    parser.add_argument('--doc_keys', type=str, required=True, help='the dataset(s) to create for processing, e.g. "title;abstract:intro;text"')
    parser.add_argument('--cranfield_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)), help='the dataset(s) to create for processing, e.g. "title;abstract:intro;text"')
    parser.add_argument('--input_dir', type=str, default=os.path.join(os.path.dirname(__file__), '..'), help='location of dataset(s)')
    args = parser.parse_args()

    if args.variant_file and not os.path.isfile(args.variant_file):
        raise Exception('variant file does not exist: {0}'.format(args.variant_file))

    t_start = time.time()
    for run_type in str(args.run_type).split(';'):

        variants = load_variants(args.variant_file) if args.variant_file else []

        query_keys = str(args.query_keys).split(';')  # query, question, narrative
        queries = load_queries(run_type, args.input_dir)
        write_queries(queries, variants, query_keys, run_type)
        del queries

        doc_keys = [s.split(':') for s in str(args.query_keys).split(';')]  # title, abstract, intro, text
        docs = load_docs(run_type, args.input_dir)
        doc_list = list(docs['uid'].keys())
        gen_dat(docs, doc_list, variants, doc_keys, run_type, args.cranfield_dir)
        del docs

        if run_type == 'train':
            qrels = load_qrels(run_type, args.input_dir)
            gen_qrels(doc_list, qrels, args.cranfield_dir)
            del qrels

    # expected run time x seconds
    print('script ran in {} seconds'.format(time.time()-t_start))

