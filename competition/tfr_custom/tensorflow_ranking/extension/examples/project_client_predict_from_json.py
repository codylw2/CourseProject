# project_client_predict_from_json.py
#
# A short end-to-end example of doing prediction for ranking problems using TFR-Bert
#
# Cobbled together by Peter Jansen based on:
#
# https://github.com/tensorflow/ranking/issues/189 by Alexander Zagniotov
# https://colab.research.google.com/github/tensorflow/ranking/blob/master/tensorflow_ranking/examples/handling_sparse_features.ipynb#scrollTo=eE7hpEBBykVS
# https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/extension/examples/tfrbert_example_test.py
# and other documentation...

from absl import flags
import tensorflow as tf
import tensorflow_ranking as tfr
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import input_pb2
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict

from tensorflow_ranking.extension import tfrbert
from functools import reduce
import json
import copy
import argparse
import time

import sys
import traceback

from official.modeling import activations
from official.nlp import optimization
from official.nlp.bert import configs
from official.nlp.bert import tokenization
from official.nlp.modeling import networks as tfmodel_networks
from tensorflow_ranking.python.keras import network as tfrkeras_network
from tensorflow_serving.apis import input_pb2

from multiprocessing import Pool
from functools import partial
from copy import deepcopy
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

    def convert_to_elwc(self, context, examples, labels, label_name):
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

        elwc = input_pb2.ExampleListWithContext()
        for example, label in zip(examples, labels):
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
            # print(feature)
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            elwc.examples.append(tf_example)
            # print(elwc)

        return elwc

    def get_max_seq_length(self):
        return self._bert_max_seq_length


class TFRBertUtilJSON(object):

    def __init__(self, TFRBertUtil):
        self.TFRBertUtilHelper = TFRBertUtil

    # Conversion function for converting easily-read JSON into TFR-Bert's ELWC format, and exporting to file
    # In: filename of JSON with ranking problems (see example json files for output)
    # Out: creates TFrecord output file, also returns list of ranking problems read in from JSON
    def convert_json_to_elwc_export(self, filenameQueryJsonIn, query_key, filenameDocJsonIn, filenameTFRecordOut):
        # Step 1: Convert JSON to ELWC
        (listToRank, listJsonRaw) = self.convert_json_to_elwc(filenameQueryJsonIn, query_key, filenameDocJsonIn, '',
                                                              False)

        # Step 2: Save ELWC to file
        try:
            with tf.io.TFRecordWriter(filenameTFRecordOut) as writer:
                for example in listToRank:
                    writer.write(example.SerializeToString())
        except:
            print("convert_json_to_elwc_export: error writing ELWC file (filename = " + filenameTFRecordOut + ")")
            exit(1)

        # Step 3: Also return ranking problem in JSON format, for use in scoring/exporting
        return listJsonRaw

    def create_chunks(self, query, title, text, max_size):
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
    def convert_json_to_elwc(self, filenameQueryJsonIn, query_key, filenameDocJsonIn, filenameJsonOut, rank=True):

        # Create an instance of the TFRBert client, to request predictions from the Tensorflow Serving model server
        tfrBertClient = TFRBertClient(grpcChannel="localhost:8500", modelName="tfrbert",
                                      servingSignatureName="serving_default", timeoutInSecs=6000)
        # tfrBertClient = TFRBertClient(grpcChannel="0.0.0.0:8500", modelName="tfrbert", servingSignatureName="serving_default", timeoutInSecs=6000)

        listToRank = list()

        # Step 1: Load JSON file
        try:
            with open(filenameQueryJsonIn) as query_file, open(filenameDocJsonIn) as doc_file:
                # Load whole JSON file
                queries = json.load(query_file)
                documents = json.load(doc_file)
                docs_at_once = 550
                docRel = 0  # required value, not used in ranking

                uid_list = documents['uid'].keys()
                last_doc = list(uid_list)[-1]
                num_docs = len(uid_list)

                ranking_problem = {
                    'q_idx': 0,
                    'documents': list()
                }
                if os.path.exists(filenameJsonOut):
                    with open(filenameJsonOut) as results_file:
                        ranking_results = json.load(results_file)
                else:
                    ranking_results = dict()

                for q_idx, query in queries.items():
                    t_start = time.time()
                    labels = list()
                    docTexts = list()
                    doc_count = 0
                    chunk_count = 0

                    queryText = query[query_key]
                    ranking_problem['q_idx'] = q_idx

                    pred_count = 0

                    for d_uid in uid_list:
                        doc_count += 1
                        doc = documents['uid'][d_uid]

                        docChunks = self.create_chunks(queryText, doc['title_tokens'], doc['tokens'],
                                                       self.TFRBertUtilHelper.get_max_seq_length())

                        for c_idx, chunk in enumerate(docChunks):
                            chunk_count += 1

                            labels.append(docRel)
                            docTexts.append(chunk)
                            ranking_problem['documents'].append(
                                {
                                    'd_uid': d_uid,
                                    'chunk': c_idx
                                }
                            )

                            if chunk_count > docs_at_once or (d_uid == last_doc and c_idx == len(docChunks) - 1):
                                pred_count += 1
                                percentCompleteStr = "{:.2f}".format(float(doc_count) * 100 / float(num_docs))
                                print("Query {} Predicting ({}%)".format(q_idx, percentCompleteStr))

                                elwcOut = self.TFRBertUtilHelper.convert_to_elwc(queryText, docTexts, labels,
                                                                                 label_name="relevance")

                                # Generate predictions for each ranking problem in the list of ranking problems in the JSON file
                                rankingProblemsOut = tfrBertClient.generatePredictionsList([elwcOut], [ranking_problem])

                                for prob in rankingProblemsOut:
                                    if prob['q_idx'] not in ranking_results.keys():
                                        ranking_results[prob['q_idx']] = prob['documents']
                                    else:
                                        # print(type(ranking_results[prob['q_idx']]))
                                        # print(ranking_results[prob['q_idx']])
                                        ranking_results[prob['q_idx']].extend(prob['documents'])

                                chunk_count = 0
                                labels = list()
                                docTexts = list()
                                ranking_problem['documents'] = list()

                    # Export ranked results to JSON file
                    tfrBertClient.exportRankingOutput(filenameJsonOut, ranking_results)

                    print('ran query {} in {} seconds'.format(q_idx, str(time.time() - t_start)))

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("convert_json_to_elwc_export: error loading JSON file (filename = " + filenameQueryJsonIn + ")")
            exit(1)

        # tfrBertClient.convert_scores_to_predictions(filenameJsonOut)


class TFRBertClient(object):
    # Default/Example values
    # self.grpcChannel = "0.0.0.0:8500"                   # from the Tensorflow Serving server
    # self.modelName = "tfrbert"
    # self.servingSignatureName = "serving_default"       # 'serving_default' instead of 'predict', as per saved_model_cli tool (see https://medium.com/@yuu.ishikawa/how-to-show-signatures-of-tensorflow-saved-model-5ac56cf1960f )
    # self.timeoutInSecs = 3

    def __init__(self, grpcChannel, modelName, servingSignatureName, timeoutInSecs):
        self.grpcChannel = grpcChannel
        self.modelName = modelName
        self.servingSignatureName = servingSignatureName
        self.timeoutInSecs = timeoutInSecs

        # Send a gRPC request to the Tensorflow Serving model server to generate predictions for a single ranking problem

    # Based on https://github.com/tensorflow/ranking/issues/189
    def generatePredictions(self, rankingProblemELWC, rankingProblemJSONIn):
        # Make a deep copy of the ranking problem
        rankingProblemJSON = copy.deepcopy(rankingProblemJSONIn)

        # Pack problem
        example_list_with_context_proto = rankingProblemELWC.SerializeToString()
        tensor_proto = tf.make_tensor_proto(example_list_with_context_proto, dtype=tf.string, shape=[1])

        # Set up request to prediction server
        request = predict_pb2.PredictRequest()
        request.inputs['input_ranking_data'].CopyFrom(tensor_proto)
        request.model_spec.signature_name = self.servingSignatureName
        request.model_spec.name = self.modelName

        channel = grpc.insecure_channel(self.grpcChannel)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # Make prediction request and get response
        grpc_response = stub.Predict(request, self.timeoutInSecs)
        unpacked_grpc_response = MessageToDict(grpc_response, preserving_proto_field_name=True)

        # Add model's ranking scores to each document
        try:
            docScores = unpacked_grpc_response['outputs']['output']['float_val']
            for docIdx in range(0, len(rankingProblemJSON['documents'])):
                rankingProblemJSON['documents'][docIdx]['score'] = docScores[docIdx]
        except Exception as e:
            print(unpacked_grpc_response)
            print(docScores)
            print(len(docScores))
            print(len(rankingProblemJSON['documents']))
            raise e

        # Sort documents in descending order based on docScore
        # rankingProblemJSON['documents'].sort(key=lambda x: x['score'], reverse=True)

        # (DEBUG) Print out ranking problem with scores added to documents
        # print(rankingProblemJSON)

        # Return ranking problem with document scores added
        return rankingProblemJSON

    # In: Parallel lists of ranking problems in ELWC and JSON format
    # Out: list of ranking problems in JSON format, with document scores from the model added, and documents sorted in descending order based on docScores.
    def generatePredictionsList(self, rankingProblemsELWC, rankingProblemsJSON):
        rankingProblemsOut = []
        # Iterate through each ranking problem, generating document scores
        for idx in range(0, len(rankingProblemsELWC)):
            rankingProblemsOut.append(self.generatePredictions(rankingProblemsELWC[idx], rankingProblemsJSON[idx]))

        return rankingProblemsOut

    def exportRankingOutput(self, filenameJSONOut, rankingProblemOutputJSON):
        print(" * exportRankingOutput(): Exporting scores to JSON (" + filenameJSONOut + ")")

        # for key, val in rankingProblemOutputJSON.items():
        #     val.sort(key=lambda x: x['score'], reverse=True)
        #     rankingProblemOutputJSON[key] = val[:1000]

        # Output JSON to file
        with open(filenameJSONOut, 'w') as outfile:
            json.dump(rankingProblemOutputJSON, outfile)

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


def main():
    # Get start time of execution
    startTime = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, required=True, help="/path/to/bert_model/vocab.txt")
    parser.add_argument("--sequence_length", type=int, required=True, help="typically 128, 256, 512")
    parser.add_argument("--query_file", type=str, required=True, help="JSON input filename (e.g. train.json)")
    parser.add_argument("--query_key", type=str, required=True, help="")
    parser.add_argument("--doc_file", type=str, required=True, help="JSON input filename (e.g. train.json)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="JSON output filename (e.g. train.scoresOut.json)")
    parser.add_argument("--do_lower_case", action="store_true", help="Set for uncased models, otherwise do not include")

    args = parser.parse_args()
    print(" * Running with arguments: " + str(args))

    # Console output
    print(" * Generating predictions for JSON ranking problems (filename: " + args.query_file + ")")

    # Create helpers
    bert_helper = create_tfrbert_util_with_vocab(args.sequence_length, args.vocab_file, args.do_lower_case)
    bert_helper_json = TFRBertUtilJSON(bert_helper)

    # Convert the JSON of input ranking problems into ELWC
    bert_helper_json.convert_json_to_elwc(args.query_file, args.query_key, args.doc_file, args.output_file)

    # Display total execution time
    print(" * Total execution time: " + secondsToStr(time.time() - startTime))


if __name__ == "__main__":
    main()