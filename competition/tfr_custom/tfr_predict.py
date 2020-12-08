import tensorflow as tf
import json
import copy
import argparse
import time
import traceback
import sys

from functools import reduce
from official.nlp import optimization
from official.nlp.bert import tokenization
from tensorflow_serving.apis import input_pb2

import os


class TFRBertUtil(object):
    """Class that defines a set of utility functions for Bert."""

    def __init__(self, bert_config_file, bert_init_ckpt, bert_max_seq_length,
                 bert_vocab_file=None, do_lower_case=None):
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

    def convert_to_elwc(self, context, examples, list_size, q_idx, uid_list):
      """Converts a <context, example list> pair to an ELWC example.

      Args:
        context: (str) raw text for a context (aka. query).
        examples: (list) raw texts for a list of examples (aka. documents).
        label_name: (str) name of the label in the ELWC example.

      Returns:
        A tensorflow.serving.ExampleListWithContext example containing the
        `input_ids`, `input_masks`, `segment_ids` and `label_id` fields.
      """

      self._truncate_tokens(context, self._bert_max_seq_length)
      context_tokens = [s.encode('utf-8') for s in context]

      elwc_list = list()
      ranking_list = list()
      zip_list = list(zip(examples, uid_list))
      for i in range(0, len(zip_list), list_size):
          elwc = input_pb2.ExampleListWithContext()
          context_feature = {
              'query_tokens':
                  tf.train.Feature(bytes_list=tf.train.BytesList(value=context_tokens))
          }
          elwc.context.CopyFrom(tf.train.Example(features=tf.train.Features(feature=context_feature)))

          ranking_problem = {
              'q_idx': q_idx,
              'documents': list()
          }

          for example_tokens, uid in zip_list[i:i+list_size]:
              self._truncate_tokens(example_tokens, self._bert_max_seq_length)
              doc_tokens = [s.encode('utf-8') for s in example_tokens]
              feature = {
                  'document_tokens':
                      tf.train.Feature(bytes_list=tf.train.BytesList(value=doc_tokens))
              }
              tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
              elwc.examples.add().CopyFrom(tf_example)

              ranking_problem['documents'].append(
                  {
                      'd_uid': uid[0],
                      'chunk': uid[1]
                  }
              )

          elwc_list.append(elwc)
          ranking_list.append(ranking_problem)

      return elwc_list, ranking_list

    def get_max_seq_length(self):
        return self._bert_max_seq_length


class TFRBertUtilJSON(object):
    def __init__(self, TFRBertUtil, modelPath):
        self.TFRBertUtilHelper = TFRBertUtil
        self.model_path = modelPath

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

    def convert_json_to_elwc(self, filenameQueryJsonIn, query_key, filenameDocJsonIn, filenameJsonOut, docsAtOnce, rerank_file):

        previous_pred = dict()
        with open(rerank_file, 'r') as f_rerank:
            for line in f_rerank.readlines():
                q_idx, uid, score = line.split()
                if int(q_idx) not in previous_pred.keys():
                    previous_pred[int(q_idx)] = list()
                previous_pred[int(q_idx)].append(uid)

        tfrBertClient = TFRBertClient(servingSignatureName='predict', modelPath=self.model_path)

        try:
            with open(filenameQueryJsonIn) as query_file, open(filenameDocJsonIn) as doc_file:
                # Load whole JSON file
                queries = json.load(query_file)
                documents = json.load(doc_file)
                docs_at_once = docsAtOnce

                if os.path.exists(filenameJsonOut):
                    with open(filenameJsonOut) as results_file:
                        ranking_results = json.load(results_file)
                else:
                    ranking_results = dict()

                max_q_idx = max([int(q) for q in ranking_results.keys()]) if ranking_results else None

                for q_idx, query in queries.items():
                    t_start = time.time()
                    docTexts = list()
                    doc_count = 0
                    chunk_count = 0

                    queryText = query[query_key]
                    d_uid_list = list()

                    pred_count = 0

                    uid_list = previous_pred[int(q_idx)]
                    last_doc = list(uid_list)[-1]
                    num_docs = len(uid_list)

                    if max_q_idx and int(q_idx) <= max_q_idx:
                        continue

                    for d_uid in uid_list:
                        doc_count += 1
                        doc = documents['uid'][d_uid]

                        text_tokens = list()
                        text_tokens.extend(doc['abstract_tokens'])
                        text_tokens.extend(doc['intro_tokens'])
                        docChunks = self.create_chunks('', doc['title_tokens'], text_tokens,
                                                       self.TFRBertUtilHelper.get_max_seq_length(), 1)

                        for c_idx, chunk in enumerate(docChunks):
                            chunk_count += 1

                            docTexts.append(chunk)
                            d_uid_list.append([d_uid, c_idx])

                            if chunk_count >= docs_at_once or (d_uid == last_doc and c_idx == len(docChunks) - 1):
                                pred_count += 1
                                percentCompleteStr = "{:.2f}".format(float(doc_count) * 100 / float(num_docs))
                                print("Query {} Predicting ({}%)".format(q_idx, percentCompleteStr))

                                elwcOut, ranking_problem = self.TFRBertUtilHelper.convert_to_elwc(queryText, docTexts, docs_at_once, q_idx, d_uid_list)

                                # Generate predictions for each ranking problem in the list of ranking problems in the JSON file
                                rankingProblemsOut = tfrBertClient.generatePredictionsList(elwcOut, ranking_problem)

                                for prob in rankingProblemsOut:
                                    if prob['q_idx'] not in ranking_results.keys():
                                        ranking_results[prob['q_idx']] = prob['documents']
                                    else:
                                        ranking_results[prob['q_idx']].extend(prob['documents'])

                                chunk_count = 0
                                docTexts = list()
                                d_uid_list = list()

                    # Export ranked results to JSON file
                    tfrBertClient.exportRankingOutput(filenameJsonOut, ranking_results)

                    print('ran query {} in {} seconds'.format(q_idx, str(time.time() - t_start)))

            tfrBertClient.exportRankingOutput(rerank_file, ranking_results)

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("convert_json_to_elwc_export: error loading JSON file (filename = " + filenameQueryJsonIn + ")")
            exit(1)


class TFRBertClient(object):
    def __init__(self, servingSignatureName, modelPath):
        self.servingSignatureName = servingSignatureName
        self.predict_fn = tf.saved_model.load(modelPath).signatures[servingSignatureName]

    def _get_latest_model(self, model_path):
        return sorted([int(d) for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))], reverse=True)[0]

    def generatePredictions(self, rankingProblemELWC, rankingProblemJSONIn):
        rankingProblemJSON = copy.deepcopy(rankingProblemJSONIn)

        rankingProblemTensor = tf.convert_to_tensor(rankingProblemELWC.SerializeToString(), dtype=tf.string)
        response = self.predict_fn(tf.expand_dims(rankingProblemTensor, axis=0))
        docScores = response['output'].numpy().tolist()[0]
        for docIdx in range(0, len(rankingProblemJSON['documents'])):
            rankingProblemJSON['documents'][docIdx]['score'] = docScores[docIdx]

        return rankingProblemJSON

    def generatePredictionsList(self, rankingProblemsELWC, rankingProblemsJSON):
        rankingProblemsOut = list()
        for idx in range(0, len(rankingProblemsELWC)):
            rankingProblemsOut.append(self.generatePredictions(rankingProblemsELWC[idx], rankingProblemsJSON[idx]))
        return rankingProblemsOut

    def exportRankingOutput(self, rerank_file, ranking_results):
        print(" * exportRankingOutput(): Exporting scores to predictions file (" + rerank_file + ")")

        with open(rerank_file, 'w') as f_txt:
            for q_idx, all_docs in ranking_results.items():
                all_docs.sort(key=lambda x: x['score'], reverse=True)
                for doc in all_docs[:1000]:
                    uid = doc['d_uid']
                    score = doc['score']
                    f_txt.write('{} {} {}\n'.format(q_idx, uid, score))

        return

    def convert_scores_to_predictions(self, filenameJSONOut):
        if os.path.isfile(filenameJSONOut):
            with open(filenameJSONOut, 'r') as f_json:
                scores = json.load(f_json)

            predict_dir = os.path.dirname(os.path.abspath(filenameJSONOut))
            with open(os.path.join(predict_dir, 'predictions.txt'), 'w') as f_predict:
                for key, val in scores.items():
                    val.sort(key=lambda x: x['score'], reverse=True)
                    scores[key] = val[:1000]

                    for doc in val:
                        f_predict.write('{} {} {}\n'.format(key, doc['d_uid'], doc['score']))
        else:
            raise Exception('score file not found: {0}'.format(filenameJSONOut))


# Adapted from TfrBertUtilTest (tfrbert_test.py)
# Creates a TFRBertUtil object, primarily used to convert ranking problems from plain text to the BERT representation packed into the ELWC format
# used by TFR-Bert.
def create_tfrbert_util_with_vocab(bertMaxSeqLength, bertVocabFile, do_lower_case):
    return TFRBertUtil(
        bert_config_file=None,
        bert_init_ckpt=None,
        bert_max_seq_length=bertMaxSeqLength,
        bert_vocab_file=bertVocabFile,
        do_lower_case=do_lower_case)


# Converts a Time to a human-readable format
# from https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution
def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
           reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                  [(t * 1000,), 1000, 60, 60])


def find_latest_model(model_base):
    saved_model_dir = os.path.join(model_base, 'export', 'saved_model_exporter')
    saved_models = [int(i) for i in os.listdir(saved_model_dir)]
    if not saved_models:
        raise Exception('no models to load: {0}'.format(saved_models))
    model_path = os.path.join(saved_model_dir, str(sorted(saved_models, reverse=True)[0]))
    return model_path


def main():
    # Get start time of execution
    startTime = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, required=True, help="/path/to/bert_model/vocab.txt")
    parser.add_argument("--sequence_length", type=int, required=True, help="typically 128, 256, 512")
    parser.add_argument("--query_file", type=str, required=True, help="JSON input filename (e.g. queries.json)")
    parser.add_argument("--query_key", type=str, required=True, help="")
    parser.add_argument("--doc_file", type=str, required=True, help="JSON input filename (e.g. docs.json)")
    parser.add_argument("--output_file", type=str, required=True, help="JSON output filename (e.g. test.scoresOut.json)")
    parser.add_argument("--model_path", type=str, required=True, help='')
    parser.add_argument("--docs_at_once", type=int, default=500, help='')
    parser.add_argument("--rerank_file", type=str, required=True, help='')
    parser.add_argument("--do_lower_case", action="store_true", help="Set for uncased models, otherwise do not include")

    args = parser.parse_args()
    print(" * Running with arguments: " + str(args))

    # Console output
    print(" * Generating predictions for JSON ranking problems (filename: " + args.query_file + ")")

    model_path = find_latest_model(args.model_path)

    # Create helpers
    bert_helper = create_tfrbert_util_with_vocab(args.sequence_length, args.vocab_file, args.do_lower_case)
    bert_helper_json = TFRBertUtilJSON(bert_helper, model_path)

    # Convert the JSON of input ranking problems into ELWC
    bert_helper_json.convert_json_to_elwc(args.query_file, args.query_key, args.doc_file, args.output_file, int(args.docs_at_once), args.rerank_file)

    # Display total execution time
    print(" * Total execution time: " + secondsToStr(time.time() - startTime))


if __name__ == "__main__":
    main()