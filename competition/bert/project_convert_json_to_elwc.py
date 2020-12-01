# project_convert_json_to_elwc.py

from tensorflow_ranking.extension import tfrbert
import argparse
import copy
import json
import os
import sys
import tensorflow as tf
import traceback

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

import project_client_predict_from_json

from multiprocessing import Pool


def load_json(json_path):
    if os.path.isfile(json_path):
        with open(json_path, 'r') as f_json:
            j_dict = json.load(f_json)
            return j_dict
    else:
        raise Exception('json file does not exist: {0}'.format(json_path))


def dump_irrelevant_docs(qrels, docs):
    query_uids = set()
    for key, val in qrels.items():
        query_uids.update(val.keys())
    doc_uids = set(docs['uid'].keys())
    for uid in doc_uids.difference(query_uids):
        docs.pop(uid, None)
    return docs


class TFRBertUtilJSON(object):

    def __init__(self, TFRBertUtil):
        self.TFRBertUtilHelper = TFRBertUtil

    # Conversion function for converting easily-read JSON into TFR-Bert's ELWC format, and exporting to file
    # In: filename of JSON with ranking problems (see example json files for output)
    # Out: creates TFrecord output file, also returns list of ranking problems read in from JSON
    def convert_json_to_elwc_export(self, filenameQueryJsonIn, filenameQueryRelJsonIn, query_key, filenameDocJsonIn, filenameTrainOut, filenameEvalOut):
        if not os.path.isdir(os.path.dirname(filenameTrainOut)):
            os.makedirs(os.path.dirname(filenameTrainOut))

        if not os.path.isdir(os.path.dirname(filenameEvalOut)):
            os.makedirs(os.path.dirname(filenameEvalOut))

        # Step 1: Convert JSON to ELWC
        (trainELWCOut, evalELWCOut) = self.convert_json_to_elwc(filenameQueryJsonIn, filenameQueryRelJsonIn, query_key, filenameDocJsonIn)

        # Step 2: Save ELWC to file
        try:
            with tf.io.TFRecordWriter(filenameTrainOut) as writer:
                for example in trainELWCOut:
                    writer.write(example.SerializeToString())
                del trainELWCOut
        except:
            print("convert_json_to_elwc_export: error writing ELWC file (filename = " + filenameTrainOut + ")")
            exit(1)

        # Step 2: Save ELWC to file
        try:
            with tf.io.TFRecordWriter(filenameEvalOut) as writer:
                for example in evalELWCOut:
                    writer.write(example.SerializeToString())
                del evalELWCOut
        except:
            print("convert_json_to_elwc_export: error writing ELWC file (filename = " + filenameEvalOut + ")")
            exit(1)

        return

    def create_chunks(self, query, title, text, max_size):
        if not text and title:
            text = title.copy()
            title = []

        chunk_size = max_size - len(query) - len(title) - 3
        if chunk_size < 0:
            title_chnk = int((max_size - len(query) - 3) / 2)
            title = title[:title_chnk]
            chunk_size = max_size - len(query) - len(title) - 3

        ln_txt = len(text)
        overlap = 50

        text_chunks = list()
        cur_idx = 0
        end_idx = 0
        while end_idx < ln_txt:
            if cur_idx > 50 and (ln_txt - cur_idx) < 50:
                print(f'{cur_idx} : {ln_txt}')
                cur_idx -= 50

            end_idx = cur_idx + chunk_size
            text_chunks.append(title + text[cur_idx:end_idx])
            cur_idx += chunk_size - overlap

        return text_chunks

    # Conversion function for converting easily-read JSON into TFR-Bert's ELWC format
    # In: JSON filename
    # Out: List of ELWC records, list of original JSON records
    def convert_json_to_elwc(self, filenameQueryJsonIn, filenameQueryRelJsonIn, query_key, filenameDocJsonIn):
        trainELWCOut = list()
        evalELWCOut = list()

        # Step 1: Load JSON file
        try:
            print('loading queries...')
            queries = load_json(filenameQueryJsonIn)
            print('loading qrels...')
            qrels = load_json(filenameQueryRelJsonIn)
            print('loading docs...')
            docs = load_json(filenameDocJsonIn)

            print('dropping irrelevant docs...')
            docs = dump_irrelevant_docs(qrels, docs)

            print('generating elwc...')
            for q_idx, query in queries.items():
                queryText = query[query_key]

                labels = []
                docTexts = []

                for doc_uid, relevance in qrels[q_idx].items():
                    doc = docs['uid'][doc_uid]
                    docChunks = self.create_chunks(queryText, doc['title_tokens'], doc['tokens'],
                                                   self.TFRBertUtilHelper.get_max_seq_length())
                    docTexts.extend(docChunks)
                    labels.extend([int(relevance) for _ in docChunks])

                elwcOut = self.TFRBertUtilHelper.convert_to_elwc(queryText, docTexts, labels, label_name="relevance")

                if int(q_idx) % 2 == 0:
                    trainELWCOut.append(elwcOut)
                else:
                    evalELWCOut.append(elwcOut)

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("convert_json_to_elwc_export: error loading JSON file (filename = " + filenameQueryJsonIn + ")")
            exit(1)

        return trainELWCOut, evalELWCOut


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, required=True, help="/path/to/bert_model/vocab.txt")
    parser.add_argument("--sequence_length", type=int, required=True, help="typically 128, 256, 512")
    parser.add_argument("--query_file", type=str, required=True, help="JSON input filename (e.g. train.json)")
    parser.add_argument("--qrel_file", type=str, required=True, help="JSON input filename (e.g. train.json)")
    parser.add_argument("--query_key", type=str, required=True, help="")
    parser.add_argument("--doc_file", type=str, required=True, help="JSON input filename (e.g. train.json)")
    parser.add_argument("--output_train_file", type=str, required=True, help="ELWC TFrecord filename (e.g. train.elwc.tfrecord)")
    parser.add_argument("--output_eval_file", type=str, required=True, help="ELWC TFrecord filename (e.g. eval.elwc.tfrecord)")
    parser.add_argument("--do_lower_case", action="store_true", help="Set for uncased models, otherwise do not include")

    args = parser.parse_args()
    print(" * Running with arguments: " + str(args))

    # Create helpers
    bert_helper = project_client_predict_from_json.create_tfrbert_util_with_vocab(args.sequence_length, args.vocab_file, args.do_lower_case)
    bert_helper_json = TFRBertUtilJSON(bert_helper)

    # User output
    print("Utility to convert between JSON and ELWC for TFR-Bert")
    print("")

    # Perform conversion of ranking problemsJSON to ELWC
    bert_helper_json.convert_json_to_elwc_export(args.query_file, args.qrel_file, args.query_key, args.doc_file, args.output_train_file, args.output_eval_file)

    print("Success.")


if __name__ == "__main__":
    main()
