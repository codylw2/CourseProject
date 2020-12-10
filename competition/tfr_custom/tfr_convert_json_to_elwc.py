# tfr_convert_json_to_elwc.py

import argparse
import copy
import json
import os
import sys
import tensorflow as tf
import traceback

from official.nlp.bert import tokenization
from tensorflow_serving.apis import input_pb2


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


def create_tfrbert_util_with_vocab(bertMaxSeqLength, bertVocabFile, do_lower_case):
    return TFRBertUtil(
            bert_config_file=None,
            bert_init_ckpt=None,
            bert_max_seq_length=bertMaxSeqLength,
            bert_vocab_file=bertVocabFile,
            do_lower_case=do_lower_case)


class TFRBertUtilJSON(object):

    def __init__(self, TFRBertUtil):
        self.TFRBertUtilHelper = TFRBertUtil

    # Conversion function for converting easily-read JSON into TFR-Bert's ELWC format, and exporting to file
    # In: filename of JSON with ranking problems (see example json files for output)
    # Out: creates TFrecord output file, also returns list of ranking problems read in from JSON
    def convert_json_to_elwc_export(self, filenameQueryJsonIn, filenameQueryRelJsonIn, query_key, filenameDocJsonIn, filenameTrainOut, filenameEvalOut, list_size):
        if not os.path.isdir(os.path.dirname(filenameTrainOut)):
            os.makedirs(os.path.dirname(filenameTrainOut))

        if not os.path.isdir(os.path.dirname(filenameEvalOut)):
            os.makedirs(os.path.dirname(filenameEvalOut))

        # Step 1: Convert JSON to ELWC
        (trainELWCOut, evalELWCOut) = self.convert_json_to_elwc(filenameQueryJsonIn, filenameQueryRelJsonIn, query_key, filenameDocJsonIn, list_size)

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

    def create_chunks(self, query, title, text, max_size, max_chunks=None):
        if not text and title:
            text = title.copy()
            title = []

        chunk_size = max_size - len(query) - len(title)
        if chunk_size < 0:
            title_chnk = int((max_size - len(query)) / 2)
            title = title[:title_chnk]
            chunk_size = max_size - len(query) - len(title)

        ln_txt = len(text)
        overlap = 50

        text_chunks = list()
        cur_idx = 0
        end_idx = 0
        while end_idx < ln_txt:
            if cur_idx > 50 and (ln_txt - cur_idx) < 50:
                # print(f'{cur_idx} : {ln_txt}')
                cur_idx -= 50

            end_idx = cur_idx + chunk_size
            text_chunks.append(title + text[cur_idx:end_idx])
            cur_idx += chunk_size - overlap

        if max_chunks:
            return text_chunks[:max_chunks]
        else:
            return text_chunks

    # Conversion function for converting easily-read JSON into TFR-Bert's ELWC format
    # In: JSON filename
    # Out: List of ELWC records, list of original JSON records
    def convert_json_to_elwc(self, filenameQueryJsonIn, filenameQueryRelJsonIn, query_key, filenameDocJsonIn, list_size):
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
                queryTokens = query[query_key]

                labels = list()
                docTokens = list()

                for doc_uid, relevance in qrels[q_idx].items():
                    doc = docs['uid'][doc_uid]
                    text_tokens = list()
                    text_tokens.extend(doc['abstract_tokens'])
                    text_tokens.extend(doc['intro_tokens'])
                    docChunks = self.create_chunks('', doc['title_tokens'], text_tokens, self.TFRBertUtilHelper.get_max_seq_length(), 1)
                    docTokens.extend(docChunks)
                    labels.extend([int(relevance) for _ in docChunks])

                elwcOutList = self.TFRBertUtilHelper.convert_to_elwc(queryTokens, docTokens, labels, list_size=list_size)

                for elwc_idx, elwcOut in enumerate(elwcOutList):
                    if int(elwc_idx) % 2 == 0:
                        evalELWCOut.append(elwcOut)
                    else:
                        trainELWCOut.append(elwcOut)

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("convert_json_to_elwc_export: error loading JSON file (filename = " + filenameQueryJsonIn + ")")
            exit(1)

        return trainELWCOut, evalELWCOut


class TFRBertUtil(object):
  """Class that defines a set of utility functions for Bert."""

  def __init__(self, bert_config_file, bert_init_ckpt, bert_max_seq_length, bert_vocab_file=None, do_lower_case=None):
    """Constructor.

    Args:
      bert_config_file: (string) path to Bert configuration file.
      bert_init_ckpt: (string)  path to pretrained Bert checkpoint.
      bert_max_seq_length: (int) maximum input sequence length (#words) after
        WordPiece tokenization. Sequences longer than this will be truncated,
        and shorter than this will be padded.
      bert_vocab_file (optional): (string) path to Bert vocabulary file.
      do_lower_case (optional): (bool) whether to lower case the input text.
        This should be aligned with the `vocab_file`.
    """
    self._bert_config_file = bert_config_file
    self._bert_init_ckpt = bert_init_ckpt
    self._bert_max_seq_length = bert_max_seq_length

    self._tokenizer = None
    if bert_vocab_file is not None and do_lower_case is not None:
      self._tokenizer = tokenization.FullTokenizer(
          vocab_file=bert_vocab_file, do_lower_case=do_lower_case)

  def _truncate_tokens(self, tokens, max_length):
      assert max_length > 0
      assert isinstance(max_length, int)
      if len(tokens) > max_length:
          del tokens[max_length:]

  def convert_to_elwc(self, context, examples, labels, list_size):
      """Converts a <context, example list> pair to an ELWC example.

      Args:
        context: (str) raw text for a context (aka. query).
        examples: (list) raw texts for a list of examples (aka. documents).
        labels: (list) a list of labels (int) for the `examples`.
        label_name: (str) name of the label in the ELWC example.

      Returns:
        A tensorflow.serving.ExampleListWithContext example containing the
        `input_ids`, `input_masks`, `segment_ids` and `label_id` fields.
      """
      if len(examples) != len(labels):
          raise ValueError("`examples` and `labels` should have the same size!")

      self._truncate_tokens(context, self._bert_max_seq_length)
      context_tokens = [s.encode('utf-8') for s in context]

      elwc_list = list()
      zip_list = list(zip(examples, labels))
      for i in range(0, len(zip_list), list_size):
          elwc = input_pb2.ExampleListWithContext()
          context_feature = {
              'query_tokens':
                  tf.train.Feature(bytes_list=tf.train.BytesList(value=context_tokens))
          }
          elwc.context.CopyFrom(tf.train.Example(features=tf.train.Features(feature=context_feature)))
          for example_tokens, label in zip_list[i:i+list_size]:
              self._truncate_tokens(example_tokens, self._bert_max_seq_length)
              doc_tokens = [s.encode('utf-8') for s in example_tokens]
              feature = {
                  'document_tokens':
                      tf.train.Feature(bytes_list=tf.train.BytesList(value=doc_tokens)),
                  'relevance':
                      tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
              }
              tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
              elwc.examples.add().CopyFrom(tf_example)
          elwc_list.append(elwc)

      return elwc_list

  def get_max_seq_length(self):
      return self._bert_max_seq_length


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
    parser.add_argument("--list_size", type=int, required=True, help='')
    parser.add_argument("--do_lower_case", action="store_true", help="Set for uncased models, otherwise do not include")

    args = parser.parse_args()
    print(" * Running with arguments: " + str(args))

    # Create helpers
    bert_helper = create_tfrbert_util_with_vocab(args.sequence_length, args.vocab_file, args.do_lower_case)
    bert_helper_json = TFRBertUtilJSON(bert_helper)

    # User output
    print("Utility to convert between JSON and ELWC for TFR")

    # Perform conversion of ranking problemsJSON to ELWC
    bert_helper_json.convert_json_to_elwc_export(args.query_file, args.qrel_file, args.query_key, args.doc_file,
                                                 args.output_train_file, args.output_eval_file, int(args.list_size))

    print("Success.")


if __name__ == "__main__":
    main()

