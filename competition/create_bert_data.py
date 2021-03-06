import argparse
import datetime
import os
import json
import pandas as pd
import re
import time
import xml.etree.ElementTree as ET

from official.nlp.bert import tokenization
from multiprocessing import Pool


def load_variants(input_dir):
    f_variants = os.path.join(input_dir, 'corona_variants.txt')
    if os.path.exists(f_variants):
        with open(f_variants, 'r') as f_txt:
            return [line.strip('\n') for line in f_txt.readlines()]


def parse_queries(input_dir, run_type):
    xmlfile = os.path.join(input_dir, f'{run_type}', 'queries.xml')
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    query_dict = dict()

    for query in root.findall('./topic'):
        query_dict[int(query.get('number'))] = {
            'query': query.find('query').text,
            'question': query.find('question').text,
            'narrative': query.find('narrative').text
        }

    return query_dict


def parse_qrels(input_dir, run_type):
    fname = os.path.join(input_dir, f'{run_type}', 'qrels.txt')
    qrels = dict()
    uids = set()
    with open(fname, 'r') as f_txt:
        for line in f_txt.readlines():
            q_idx, d_uid, relevance = line.strip('\n').split()
            uids.add(d_uid)

            if q_idx not in qrels.keys():
                qrels[q_idx] = dict()
            qrels[q_idx][d_uid] = int(relevance)
    return qrels, uids


def write_json(queries, output_dir, f_name, indent=False):
    with open(os.path.join(output_dir, f_name), 'w') as json_f:
        if indent:
            json.dump(queries, json_f, indent=4)
        else:
            json.dump(queries, json_f)


def process_publish_time(pub_time_str, re_date, re_year):
    if pub_time_str and re_date.match(pub_time_str):
        return datetime.datetime.strptime(pub_time_str, '%m/%d/%Y').strftime('%m/%d/%Y')
    elif pub_time_str and re_year.match(pub_time_str):
        return datetime.datetime.strptime(pub_time_str, '%Y').strftime('%m/%d/%Y')
    else:
        return ''


def determine_docs(run_type, uids, input_dir):
    metadata_csv = os.path.join(input_dir, f'{run_type}', 'documents', 'metadata.csv')
    df = pd.read_csv(metadata_csv, dtype=str)
    rows = df.to_dict('records')

    re_year = re.compile('^\d+$')
    re_date = re.compile('^\d+/\d+/\d+$')
    re_int = re.compile('^[0-9]+$')

    doc_count = 0
    doc_list = list()
    doc_dict = {'id': dict(), 'uid': dict(), 's2id': dict(), 'uid2uid': dict()}
    for idx, row in enumerate(rows):
        uid = row['uid'].strip()
        uid = doc_dict['uid2uid'].get(uid, uid)

        s2id = int(row['s2_id']) if str(row.get('s2_id', 'nan')) != 'nan' and re_int.match(str(row.get('s2_id', 'nan'))) else None
        # if s2id and s2id in doc_dict['s2id'].keys():
        #     doc_dict['s2id'][s2id].append(uid)
        #     uid_orig = uid
        #     uid = doc_dict['s2id'][s2id][0]
        #     doc_dict['uid2uid'][uid_orig] = uid
        #
        # elif s2id:
        #     doc_dict['s2id'][s2id] = [uid]

        if uids and uid not in uids:
            continue
        
        if uid not in doc_dict['uid'].keys():
            doc_dict['uid'][uid] = {
                'title': str(row['title']).replace('\r', '').replace('\n', ' ') if str(row['title']) != 'nan' else '',
                'abstract': str(row['abstract']).replace('\r', '').replace('\n', ' ') if str(row['abstract']) != 'nan' else '',
                'pmcid': row['pmcid'],
                's2id': s2id,
                'date': process_publish_time(str(row['publish_time']), re_date, re_year),
                'pdf_json_files': [d.strip() for d in row.get('pdf_json_files', '').split(';')] if str(row.get('pdf_json_files', 'nan')) != 'nan' else None,
                'pmc_json_files': row.get('pmc_json_files', None) if str(row.get('pmc_json_files', 'nan')) != 'nan' else None
            }

            doc_dict['id'][doc_count] = uid
            doc_list.append(uid)
            doc_count += 1
        else:
            if not doc_dict['uid'][uid]['abstract']:
                doc_dict['uid'][uid]['abstract'] = str(row['abstract']).replace('\r', '').replace('\n', ' ') if str(row['abstract']) != 'nan' else ''
            if not doc_dict['uid'][uid]['date']:
                doc_dict['uid'][uid]['date'] = process_publish_time(str(row['publish_time']), re_date, re_year)

            if not doc_dict['uid'][uid]['s2id']:
                doc_dict['uid'][uid]['s2id'] = s2id

            if not doc_dict['uid'][uid]['pmc_json_files'] or 'pmc' not in doc_dict['uid'][uid]['pmc_json_files']:
                doc_dict['uid'][uid]['pmc_json_files'] = row.get('pmc_json_files', None) if str(row.get('pmc_json_files', 'nan')) != 'nan' else None
                doc_dict['uid'][uid]['pmcid'] = row['pmcid']

            if doc_dict['uid'][uid]['pdf_json_files']:
                doc_dict['uid'][uid]['pdf_json_files'].extend([d.strip() for d in row.get('pdf_json_files', '').split(';')] if str(row.get('pdf_json_files', 'nan')) != 'nan' else [])
            else:
                doc_dict['uid'][uid]['pdf_json_files'] = [d.strip() for d in row.get('pdf_json_files', '').split(';')] if str(row.get('pdf_json_files', 'nan')) != 'nan' else None

    print(len(doc_list))
    print(len(doc_dict['uid'].keys()))

    return doc_dict, doc_list


def determine_text(src_list, idx=0):
    json_path = src_list[idx]

    if json_path:
        with open(json_path, 'r') as j_f:
            doc = json.load(j_f)
            text_list = list()
            abstract_list = list()
            intro_list = list()

            for section_nm in ['abstract']:
                section = doc.get(section_nm, '')
                if section:
                    if isinstance(section, list):
                        for sub_section in section:
                            abstract_list.append(sub_section['text'])
                    elif section_nm == 'ref_entries':
                        for key in section.keys():
                            abstract_list.append(section[key]['text'])
                    else:
                        abstract_list.append(section['text'])

            for section_nm in ['body_text', 'ref_entries']:
                section = doc.get(section_nm, '')
                if section:
                    if isinstance(section, list):
                        for sub_section in section:
                            if 'intro' in sub_section['section']:
                                intro_list.append(sub_section['text'])
                            elif 'abstract' in sub_section['section']:
                                abstract_list.append(sub_section['text'])
                            else:
                                text_list.append(sub_section['text'])
                    elif section_nm == 'ref_entries':
                        for key in section.keys():
                            text_list.append(section[key]['text'])
                    else:
                        if 'intro' in section['section']:
                            intro_list.append(section['text'])
                        elif 'abstract' in section['section']:
                            abstract_list.append(section['text'])
                        else:
                            text_list.append(section['text'])
            abstract = ' '.join(abstract_list)
            intro = ' '.join(intro_list)
            text = ' '.join(text_list)
    else:
        abstract = ''
        intro = ''
        text = ''

    return abstract.replace('\r', '').replace('\n', ' '), intro.replace('\r', '').replace('\n', ' '), text.replace('\r', '').replace('\n', ' '),


def populate_doc_text(args, doc_dict, doc_list, run_type):
    input_dir = args.input_dir
    text_max_len = 0
    text_max_doc = ''
    count_empty = 0

    # remove url links
    re_http = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

    comp_variants = [re.compile(v) for v in load_variants(input_dir)] if args.variant_file else []

    for uid in doc_list:
        src_list = list()
        json_path = doc_dict['uid'][uid]['pmc_json_files']
        if json_path:
            json_path = os.path.join(input_dir, f'{run_type}', 'documents', json_path)
            src_list.append(json_path)

        json_list = doc_dict['uid'][uid]['pdf_json_files']
        if json_list:
            src_list.extend([os.path.join(input_dir, f'{run_type}', 'documents', j_path) for j_path in json_list])

        abstract = ''
        intro = ''
        text = ''
        doc_idx = 0
        while src_list and doc_idx < len(src_list):
            abstract, intro, text = determine_text(src_list, doc_idx)

            if len(abstract.split()) > 3 or len(intro.split()) > 3:
                break
            else:
                text = ''
                doc_idx += 1

        # if not text and len(doc_dict['uid'][uid]['abstract'].split()) > 3:
        #     text = doc_dict['uid'][uid]['abstract']
        # elif not text:
        #     count_empty += 1
        #     doc_dict['uid'].pop(uid, None)
        #     continue

        # if len(text) > text_max_len:
        #     text_max_len = len(text)
        #     if len(text) > 15:
        #         text_max_doc = src_list[doc_idx]
        #     else:
        #         text_max_doc = src_list[doc_idx-1]

        text = re_http.sub('', text)
        for re_var in comp_variants:
            text = re_var.sub(args.variant_default, text)

        doc_dict['uid'][uid]['abstract'] = abstract if abstract and not doc_dict['uid'][uid]['abstract'] else doc_dict['uid'][uid]['abstract']
        doc_dict['uid'][uid]['intro'] = intro
        doc_dict['uid'][uid]['text'] = text
        # del(doc_dict['uid'][uid]['abstract'])
        del(doc_dict['uid'][uid]['pmcid'])
        del(doc_dict['uid'][uid]['pmc_json_files'])
        del(doc_dict['uid'][uid]['pdf_json_files'])

    # print('text_max_len: {}'.format(text_max_len))
    # print('text_max_doc: {}'.format(text_max_doc))
    # print('count_empty: {}'.format(count_empty))

    return doc_dict


def truncate_seq(tokens, max_length):
    assert max_length > 0
    assert isinstance(max_length, int)
    if len(tokens) > max_length:
        del tokens[max_length - len(tokens):]


def tokenize_query(vocab_file, query_idx, query):
    max_length = 1024
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    tokens = tokenizer.tokenize(query['query'])
    # truncate_seq(tokens, max_length)
    query['query_tokens'] = tokens
    tokens = tokenizer.tokenize(query['question'])
    # truncate_seq(tokens, max_length)
    query['question_tokens'] = tokens
    tokens = tokenizer.tokenize(query['narrative'])
    # truncate_seq(tokens, max_length)
    query['narrative_tokens'] = tokens
    return query_idx, query


def tokenize_queries(queries, vocab_file):
    procs = list()
    with Pool() as p:
        for key, val in queries.items():
            procs.append(p.apply_async(tokenize_query, (vocab_file, key, val)))

        for proc in procs:
            (key, val) = proc.get()
            queries[key] = val
    return queries


def tokenize_doc(vocab_file, doc_uid, doc):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    title_tokens = tokenizer.tokenize(doc['title'])
    doc['title_tokens'] = title_tokens

    title_tokens = tokenizer.tokenize(doc['abstract'])
    doc['abstract_tokens'] = title_tokens

    title_tokens = tokenizer.tokenize(doc['intro'])
    doc['intro_tokens'] = title_tokens

    tokens = tokenizer.tokenize(doc['text'])
    doc['text_tokens'] = tokens
    return doc_uid, doc


def tokenize_docs(doc_dict, vocab_file):
    procs = list()
    with Pool() as p:
        for key, val in doc_dict['uid'].items():
            procs.append(p.apply_async(tokenize_doc, (vocab_file, key, val)))

        for proc in procs:
            (key, val) = proc.get()
            if key and val:
                doc_dict['uid'][key] = val
    return doc_dict


if __name__ == '__main__':
    t_start = time.time()

    parser = argparse.ArgumentParser(description='Process dataset and produce json of processed data')
    parser.add_argument('--vocab_file', type=str, help='/path/to/bert_model/vocab.txt')
    parser.add_argument('--variant_file', type=str, help='/path/to/bert_model/corona_variants.txt')
    parser.add_argument('--variant_default', type=str, default='coronavirus', help='default value to replace corona variants')
    parser.add_argument('--run_type', type=str, default='train;test', help='the dataset(s) to process, e.g. "train;test"')
    parser.add_argument('--tokenize', action='store_true', default=True, help='tokenize the queries and docs')
    parser.add_argument('--input_dir', type=str, default=os.path.dirname(__file__), help='location of dataset(s)')
    parser.add_argument('--output_dir', type=str, default=os.path.dirname(__file__), help='location to put the created json files')
    parser.add_argument('--output_prefix', type=str, help='descriptop to prefix to datasets')
    args = parser.parse_args()

    if args.tokenize and not args.vocab_file:
        raise Exception('a vocabulary file is required if tokenization is to happen')
    elif args.tokenize and args.vocab_file:
        if not os.path.isfile(args.vocab_file):
            raise Exception('vocab file does not exist: {0}'.format(args.vocab_file))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for run_type in str(args.run_type).split(';'):
        file_pre = run_type
        if args.output_prefix:
            file_pre = '{0}_{1}'.format(args.output_prefix, run_type)

        print(run_type)
        queries = parse_queries(args.input_dir, run_type)
        queries = tokenize_queries(queries, args.vocab_file)
        write_json(queries, args.output_dir, f'{file_pre}_queries.json', True)
        del queries

        uids = set()
        if run_type == 'train':
            qrels, uids = parse_qrels(args.input_dir, run_type)
            write_json(qrels, args.output_dir, f'{file_pre}_qrels.json', True)
            del qrels

        doc_dict, doc_list = determine_docs(run_type, uids, args.input_dir)
        doc_dict = populate_doc_text(args, doc_dict, doc_list, run_type)
        doc_dict = tokenize_docs(doc_dict, args.vocab_file)
        write_json(doc_dict, args.output_dir, f'{file_pre}_docs.json')
        del doc_dict

    print('script ran in {} seconds'.format(time.time()-t_start))
