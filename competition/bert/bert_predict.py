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

    def create_optimizer(self, init_lr, train_steps, warmup_steps, optimizer_type="adamw"):
        """Creates an optimizer for TFR-BERT.

        Args:
          init_lr: (float) the init learning rate.
          train_steps: (int) the number of train steps.
          warmup_steps: (int) if global_step < num_warmup_steps, the learning rate
            will be `global_step / num_warmup_steps * init_lr`. See more details in
            the `tensorflow_models.official.nlp.optimization.py` file.
          optimizer_type: (string) Optimizer type, can either be `adamw` or `lamb`.
            Default to be the `adamw` (AdamWeightDecay). See more details in the
            `tensorflow_models.official.nlp.optimization.py` file.

        Returns:
          The optimizer training op.
        """

        return optimization.create_optimizer(
            init_lr=init_lr,
            num_train_steps=train_steps,
            num_warmup_steps=warmup_steps,
            optimizer_type=optimizer_type)

    def get_warm_start_settings(self, exclude):
        """Defines warm-start settings for the TFRBert ranking estimator.

        Our TFRBert ranking models will warm-start from a pre-trained Bert model.
        Here, we define the warm-start setting by excluding non-Bert parameters.

        Args:
          exclude: (string) Variable to exclude from the warm-start settings.

        Returns:
          (`tf.estimator.WarmStartSettings`) the warm-start setting for the TFRBert
          ranking estimator.
        """
        # A regular expression to exclude the variables starting with the passed-in
        # `exclude` parameter. Variables from the downloaded Bert checkpoints often
        # start with `transformer`, `pooler`, `embeddings` and etc., whereas other
        # variables specifically to the TFRBertRankingNetwork start with the `name`
        # we passed to the `TFRBertRankingNetwork` constructor. When defining the
        # warm-start settings, we exclude those non-Bert variables.
        vars_to_warm_start = "^(?!{exclude}).*$".format(exclude=exclude)
        return tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=self._bert_init_ckpt,
            vars_to_warm_start=vars_to_warm_start)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which truncates the longer sequence until
        # their combined length is no longer than max_length.
        # This makes more sense than truncating with an equal percentage of tokens
        # from each, since if a sequence is very short then each token that's
        # truncated likely contains more information than a longer sequence.
        assert max_length > 0
        assert isinstance(max_length, int)
        if len(tokens_a) + len(tokens_b) > max_length:
            # Truncation is needed.
            if (len(tokens_a) >= max_length - max_length // 2 and
                    len(tokens_b) >= max_length // 2):
                # Truncate both sequences until they have almost equal lengths and the
                # combined length is no longer than max_length
                del tokens_a[max_length - max_length // 2:]
                del tokens_b[max_length // 2:]
            elif len(tokens_a) > len(tokens_b):
                # Only truncating tokens_a would suffice
                del tokens_a[max_length - len(tokens_b):]
            else:
                # Only truncating tokens_b would suffice
                del tokens_b[max_length - len(tokens_a):]

    def _to_bert_ids(self, sent_a, sent_b=None):
        """Converts a sentence pair (sent_a, sent_b) to related Bert ids.

        This function is mostly adopted from run_classifier.convert_single_example
        in bert/run_classifier.py.

        Args:
          sent_a: (str) the raw text of the first sentence.
          sent_b: (str) the raw text of the second sentence.

        Returns:
          A tuple (`input_ids`, `input_masks`, `segment_ids`) for Bert finetuning.
        """

        if self._tokenizer is None:
            raise ValueError("Please pass both `vocab_file` and `do_lower_case` in "
                             "the BertUtil constructor to build a Bert tokenizer!")

        if sent_a is None:
            raise ValueError("`sent_a` cannot be None!")

        tokens_a = sent_a
        tokens_b = sent_b

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is
            # less than the specified length. Since the final sequence will be
            # [CLS] `tokens_a` [SEP] `tokens_b` [SEP], thus, we use `- 3`.
            self._truncate_seq_pair(tokens_a, tokens_b, self._bert_max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2".  Since there is only one
            # sentence, we don't need to account for the second [SEP].
            self._truncate_seq_pair(tokens_a, [], self._bert_max_seq_length - 2)

        # The convention in BERT for sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        #
        # The `type_ids` (aka. `segment_ids`) are used to indicate whether this is
        # the first or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        #
        # When there is only one sentence given, the sequence pair would be:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] + [0] * len(tokens_a) + [0]
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * len(tokens_b) + [1]
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        if len(input_ids) < self._bert_max_seq_length:
            padding_len = self._bert_max_seq_length - len(input_ids)
            input_ids.extend([0] * padding_len)
            input_mask.extend([0] * padding_len)
            segment_ids.extend([0] * padding_len)

        assert len(input_ids) == self._bert_max_seq_length
        assert len(input_mask) == self._bert_max_seq_length
        assert len(segment_ids) == self._bert_max_seq_length

        return input_ids, input_mask, segment_ids

    def convert_to_elwc(self, context, examples, labels, label_name, list_size, q_idx, uid_list):
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

        elwc_list = list()
        ranking_list = list()
        zip_list = list(zip(examples, labels, uid_list))
        for i in range(0, len(zip_list), list_size):

            elwc = input_pb2.ExampleListWithContext()

            ranking_problem = {
              'q_idx': q_idx,
              'documents': list()
            }

            for example, label, uid in zip_list[i:i+list_size]:
                (input_ids, input_mask, segment_ids) = self._to_bert_ids(context, example)

                feature = {
                    "input_ids":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
                    "input_mask":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=input_mask)),
                    "segment_ids":
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=segment_ids)),
                    label_name:
                        tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                elwc.examples.append(tf_example)

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

        tfrBertClient = TFRBertClient(servingSignatureName="serving_default", modelPath=self.model_path)

        try:
            with open(filenameQueryJsonIn) as query_file, open(filenameDocJsonIn) as doc_file:
                # Load whole JSON file
                queries = json.load(query_file)
                documents = json.load(doc_file)
                docs_at_once = docsAtOnce
                docRel = 0  # required value, not used in ranking
                ranking_results = dict()

                q_keys = sorted([int(q) for q in queries.keys()])
                for q_key in q_keys:
                    q_idx = str(q_key)
                    query = queries[q_idx]

                    t_start = time.time()
                    labels = list()
                    docTexts = list()
                    doc_count = 0
                    chunk_count = 0

                    queryText = query[query_key]
                    d_uid_list = list()

                    pred_count = 0

                    uid_list = previous_pred[int(q_idx)]
                    last_doc = list(uid_list)[-1]
                    num_docs = len(uid_list)

                    for d_uid in uid_list:
                        doc_count += 1
                        doc = documents['uid'][d_uid]

                        text_tokens = list()
                        text_tokens.extend(doc['abstract_tokens'])
                        text_tokens.extend(doc['intro_tokens'])
                        docChunks = self.create_chunks(queryText, doc['title_tokens'], text_tokens,
                                                       self.TFRBertUtilHelper.get_max_seq_length(), 1)

                        for c_idx, chunk in enumerate(docChunks):
                            chunk_count += 1

                            labels.append(docRel)
                            docTexts.append(chunk)
                            d_uid_list.append([d_uid, c_idx])

                            if chunk_count >= docs_at_once or (d_uid == last_doc and c_idx == len(docChunks) - 1):
                                pred_count += 1
                                percentCompleteStr = "{:.2f}".format(float(doc_count) * 100 / float(num_docs))
                                print("Query {} Predicting ({}%)".format(q_idx, percentCompleteStr))

                                elwcOutList, ranking_problems = self.TFRBertUtilHelper.convert_to_elwc(queryText, docTexts, labels, "relevance", docs_at_once, q_idx, d_uid_list)

                                # Generate predictions for each ranking problem in the list of ranking problems in the JSON file
                                rankingProblemsOut = tfrBertClient.generatePredictionsList(elwcOutList, ranking_problems)

                                for prob in rankingProblemsOut:
                                    if prob['q_idx'] not in ranking_results.keys():
                                        ranking_results[prob['q_idx']] = prob['documents']
                                    else:
                                        ranking_results[prob['q_idx']].extend(prob['documents'])

                                chunk_count = 0
                                labels = list()
                                docTexts = list()
                                d_uid_list = list()

                    print('ran query {} in {} seconds'.format(q_idx, str(time.time() - t_start)))

                    # Export ranked results to JSON file
                    if int(q_idx) % 5 == 0:
                        tfrBertClient.exportRankingOutput(filenameJsonOut, ranking_results)

                if int(q_idx) % 5 != 0:
                    tfrBertClient.exportRankingOutput(filenameJsonOut, ranking_results)

            tfrBertClient.exportRankingPredictions(rerank_file, ranking_results)

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("convert_json_to_elwc_export: error loading JSON file (filename = " + filenameQueryJsonIn + ")")
            exit(1)


class TFRBertClient(object):
    def __init__(self, servingSignatureName, modelPath):
        self.servingSignatureName = servingSignatureName
        self.predict_fn = tf.saved_model.load(modelPath).signatures['serving_default']

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

    def exportRankingOutput(self, filenameJSONOut, rankingProblemOutputJSON):
        print(" * exportRankingOutput(): Exporting scores to JSON (" + filenameJSONOut + ")")
        # Output JSON to file
        with open(filenameJSONOut, 'w') as outfile:
            json.dump(rankingProblemOutputJSON, outfile)
        return

    def exportRankingPredictions(self, rerank_file, ranking_results):
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
    saved_model_dir = os.path.join(model_base, 'export', 'latest_model')
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
    parser.add_argument("--docs_at_once", type=int, default=550, help='')
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