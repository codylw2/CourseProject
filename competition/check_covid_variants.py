import argparse
import json
import os

from multiprocessing import Pool


def load_variants(var_file):
    print('loading known variants...')
    if os.path.exists(var_file):
        with open(var_file, 'r', encoding='utf-8') as f_txt:
            variants = [line.strip('\n') for line in f_txt.readlines()]
        return set(variants)
    else:
        return set()


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


def load_docs(run_type, input_dir):
    print('loading docs...')
    f_name = '{0}_docs.json'.format(run_type)
    return load_json(os.path.join(input_dir, '..', f_name))


def determine_variants(txt, known_variants):
    found_variants = set()
    word_list = txt.split()
    for word in word_list:
        if any([x in word for x in known_variants]):
            found_variants.add(word)
    return found_variants


def process_txt(txt, known_variants):
    txt = txt.lower()
    variants = determine_variants(txt, known_variants)
    return variants


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process dataset and produce json of processed data')
    parser.add_argument('--variant_file', type=str, help='/path/to/bert_model/corona_variants.txt')
    parser.add_argument('--known_variants', type=str, default='', help='sars;ncov;covid')
    parser.add_argument('--doc_keys', type=str, required=True, help='the dataset(s) to create for processing, e.g. "title;abstract:intro;text"')
    parser.add_argument('--run_type', type=str, default='train;test', help='the dataset(s) to process, e.g. "train;test"')
    parser.add_argument('--input_dir', type=str, default=os.path.dirname(__file__), help='location to put the created json files')
    args = parser.parse_args()

    if args.variant_file and not os.path.isfile(args.variant_file):
        raise Exception('variant file does not exist: {0}'.format(args.variant_file))

    script_dir = os.path.dirname(__file__)

    known_variants = args.known_variants.split(';') if args.known_variants else []
    variants = load_variants(args.variant_file)
    for run_type in str(args.run_type).split(';'):
        queries = load_queries(run_type, args.input_dir)
        docs = load_docs(run_type, args.input_dir)

        print('launching pool...')
        procs = list()
        with Pool(12) as p:
            print('pool created, launching procs...')
            for key, val in queries.items():
                procs.append(p.apply_async(process_txt, (val['query'], )))
            del queries

            for key, val in docs['uid'].items():
                for doc_key in args.doc_keys.split(';'):
                    procs.append(p.apply_async(process_txt, (val[doc_key], known_variants)))
            del docs

            for proc in procs:
                res = proc.get()
                variants.update(res)

    with open(args.variant_file, 'w', encoding='utf-8') as f_txt:
        for word in sorted(variants):
            f_txt.write(word + '\n')
