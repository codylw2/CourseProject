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



class TFRBertUtilJSON(object):

    def __init__(self, TFRBertUtil):
        self.TFRBertUtilHelper = TFRBertUtil


    # Conversion function for converting easily-read JSON into TFR-Bert's ELWC format, and exporting to file
    # In: filename of JSON with ranking problems (see example json files for output)
    # Out: creates TFrecord output file, also returns list of ranking problems read in from JSON
    def convert_json_to_elwc_export(self, filenameQueryJsonIn, query_key, filenameDocJsonIn, filenameTFRecordOut):
        # Step 1: Convert JSON to ELWC
        (listToRank, listJsonRaw) = self.convert_json_to_elwc(filenameQueryJsonIn, query_key, filenameDocJsonIn, '', False)

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
        tfrBertClient = TFRBertClient(grpcChannel="localhost:8500", modelName="tfrbert", servingSignatureName="serving_default", timeoutInSecs=6000)
        # tfrBertClient = TFRBertClient(grpcChannel="0.0.0.0:8500", modelName="tfrbert", servingSignatureName="serving_default", timeoutInSecs=6000)

        listToRank = list()

        # Step 1: Load JSON file
        try:
            with open(filenameQueryJsonIn) as query_file, open(filenameDocJsonIn) as doc_file:
                # Load whole JSON file
                queries = json.load(query_file)
                documents = json.load(doc_file)
                docs_at_once = 500
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

                        docChunks = self.create_chunks(queryText, doc['title_tokens'], doc['tokens'], self.TFRBertUtilHelper.get_max_seq_length())

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

                            if rank and (chunk_count > docs_at_once or (d_uid == last_doc and c_idx == len(docChunks)-1)):
                                pred_count += 1
                                percentCompleteStr = "{:.2f}".format(float(doc_count) * 100 / float(num_docs))
                                print("Query {} Predicting ({}%)".format(q_idx, percentCompleteStr))

                                elwcOut = self.TFRBertUtilHelper.convert_to_elwc(queryText, docTexts, labels, label_name="relevance")

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

                    if rank:
                        # Export ranked results to JSON file
                        tfrBertClient.exportRankingOutput(filenameJsonOut, ranking_results)

                        print('ran query {} in {} seconds'.format(q_idx, str(time.time()-t_start)))

                if not rank:
                    listToRank.extend(self.TFRBertUtilHelper.convert_to_elwc(queryText, docTexts, labels, label_name="relevance"))

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("convert_json_to_elwc_export: error loading JSON file (filename = " + filenameQueryJsonIn + ")")
            exit(1)

        if rank:
            tfrBertClient.convert_scores_to_predictions(filenameJsonOut)
            return None
        else:
            return listToRank
        

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
        unpacked_grpc_response = MessageToDict(grpc_response, preserving_proto_field_name = True)

        # Add model's ranking scores to each document
        docScores = unpacked_grpc_response['outputs']['output']['float_val']
        for docIdx in range(0, len(rankingProblemJSON['documents'])):
            rankingProblemJSON['documents'][docIdx]['score'] = docScores[docIdx]

        # Sort documents in descending order based on docScore
        rankingProblemJSON['documents'].sort(key=lambda x:x['score'], reverse=True)

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
            rankingProblemsOut.append( self.generatePredictions(rankingProblemsELWC[idx], rankingProblemsJSON[idx]) )

        return rankingProblemsOut
    

    def exportRankingOutput(self, filenameJSONOut, rankingProblemOutputJSON):
        print(" * exportRankingOutput(): Exporting scores to JSON (" + filenameJSONOut + ")")

        for key, val in rankingProblemOutputJSON.items():
            val.sort(key=lambda x: x['score'], reverse=True)
            rankingProblemOutputJSON[key] = val[:1000]

        # Output JSON to file
        with open(filenameJSONOut, 'w') as outfile:
            json.dump(rankingProblemOutputJSON, outfile, indent=4)


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
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])


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
    parser.add_argument("--output_file", type=str, required=True, help="JSON output filename (e.g. train.scoresOut.json)")
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