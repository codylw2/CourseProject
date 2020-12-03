import os
import json
import re
import time
import datetime

from multiprocessing import Pool
from official.nlp.bert import tokenization


def load_queries(run_type):
    print('loading queries...')
    f_name = '{0}_queries.json'.format(run_type)
    with open(os.path.join(script_dir, '..', f_name), 'r') as json_f:
        return json.load(json_f)


def load_docs(run_type):
    print('loading docs...')
    f_name = '{0}_docs.json'.format(run_type)
    with open(os.path.join(script_dir, '..', f_name), 'r') as json_f:
        return json.load(json_f)


def load_variants():
    with open(os.path.join(script_dir, '..', 'corona_variants.txt'), 'r') as f_txt:
        return [line.strip('\n') for line in f_txt.readlines()]


def write_queries(queries, variants):
    with open('cranfield-queries.txt', 'w') as txt:
        for key in sorted([int(k) for k in queries.keys()]):
            query_text = queries[str(key)]['query']
            # query_text = ' '.join([queries[str(key)][subkey] for subkey in ['query', 'question', 'narrative']])  # query, question, narrative
            query_text, _ = update_text(query_text, None, variants)
            txt.writelines(query_text + '\n')


def update_text(text, uid, variants):
    text = text.lower()
    re_http = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
    text = re_http.sub('', text)

    # comp_variants = [re.compile(v) for v in variants]
    # for re_var in comp_variants:
    #     text = re_var.sub('covid-19', text)

    # basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=True, split_on_punc=True)
    # tokens = basic_tokenizer.tokenize(text)
    # text = ' '.join(tokens)

    return text, uid


def gen_dat(doc_dict, doc_list, variants):
    print('generating dat...')
    outfile = open(os.path.join('cranfield', 'cranfield.dat'), 'w', encoding='utf-8')
    orderfile = open(os.path.join('cranfield', 'cranfield-dat.json'), 'w', encoding='utf-8')

    procs = list()
    with Pool(12) as p:
        max_len = 0
        max_uid = ''
        min_date = datetime.datetime(2015, 1, 1)
        for uid in doc_list:

            curr_date = None
            if doc_dict['uid'][uid]['date']:
                curr_date = datetime.datetime.strptime(doc_dict['uid'][uid]['date'], '%d/%m/%Y')

            if curr_date and curr_date >= min_date:
                comb_txt = ' '.join([doc_dict['uid'][uid][key] for key in ['title', 'abstract', 'intro']]) # title, abstract, intro, text
                if len(comb_txt.split(' ')) > max_len:
                    max_len = len(comb_txt.split(' '))
                    max_uid = uid
                procs.append(p.apply_async(update_text, (comb_txt, uid, variants)))


        print('max_len: {}'.format(max_len))
        print('max_uid: {}'.format(max_uid))

        for proc_idx, proc in enumerate(procs):
            text, uid = proc.get()
            doc_dict['uid'][uid] = {'text': text}

    doc_count = 0
    for uid in doc_list[:]:
        outfile.write(doc_dict['uid'][uid]['text'] + '\n')
        # if any([s in doc_dict['uid'][uid]['text'] for s in ['coronavirus', 'sars', 'mers', 'pandemic', 'violence', 'dexamethasone']]):
        #     outfile.write(doc_dict['uid'][uid]['text'] + '\n')
        #     doc_count += 1
        # else:
        #     doc_list.remove(uid)

    print(doc_count)
    print(len(doc_list))

    json.dump({'uid_order': doc_list}, orderfile)

    outfile.close()
    orderfile.close()

    return


if __name__ == '__main__':
    t_start = time.time()
    script_dir = os.path.dirname(__file__)
    run_type = 'test'

    variants = load_variants()

    queries = load_queries(run_type)
    write_queries(queries, variants)

    docs = load_docs(run_type)
    doc_list = list(docs['uid'].keys())
    gen_dat(docs, doc_list, variants)

    # expected run time x seconds
    print('script ran in {} seconds'.format(time.time()-t_start))
