# project_convert_json_to_elwc.py

from tensorflow_ranking.extension import tfrbert
import argparse
import copy
import json
import os
import sys
import tensorflow as tf
import traceback

from official.nlp import optimization
from official.nlp.bert import tokenization
from tensorflow_serving.apis import input_pb2


script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)


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

  def convert_to_elwc(self, context, examples, labels, label_name, list_size):
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
      zip_list = list(zip(examples, labels))
      for i in range(0, len(zip_list), list_size):

          elwc = input_pb2.ExampleListWithContext()
          for example, label in zip_list[i:i+list_size]:
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
          elwc_list.append(elwc)

      return elwc_list

  def get_max_seq_length(self):
      return self._bert_max_seq_length


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
                queryText = query[query_key]

                labels = []
                docTexts = []

                for doc_uid, relevance in qrels[q_idx].items():
                    doc = docs['uid'][doc_uid]
                    text_tokens = list()
                    text_tokens.extend(doc['abstract_tokens'])
                    text_tokens.extend(doc['intro_tokens'])
                    docChunks = self.create_chunks(queryText, doc['title_tokens'], text_tokens, self.TFRBertUtilHelper.get_max_seq_length(), 1)
                    docTexts.extend(docChunks)
                    labels.extend([int(relevance) for _ in docChunks])

                elwcOutList = self.TFRBertUtilHelper.convert_to_elwc(queryText, docTexts, labels, label_name="relevance", list_size=list_size)

                for elwc_idx, elwcOut in enumerate(elwcOutList):
                    if elwc_idx == 0 or elwc_idx % 3 != 0:
                        trainELWCOut.append(elwcOut)
                    else:
                        evalELWCOut.append(elwcOut)

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("convert_json_to_elwc_export: error loading JSON file (filename = " + filenameQueryJsonIn + ")")
            exit(1)

        return trainELWCOut, evalELWCOut


# used by TFR-Bert.
def create_tfrbert_util_with_vocab(bertMaxSeqLength, bertVocabFile, do_lower_case):
    return TFRBertUtil(
            bert_config_file=None,
            bert_init_ckpt=None,
            bert_max_seq_length=bertMaxSeqLength,
            bert_vocab_file=bertVocabFile,
            do_lower_case=do_lower_case)


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
    print("Utility to convert between JSON and ELWC for TFR-Bert")
    print("")

    # Perform conversion of ranking problemsJSON to ELWC
    bert_helper_json.convert_json_to_elwc_export(args.query_file, args.qrel_file, args.query_key, args.doc_file, args.output_train_file, args.output_eval_file, args.list_size)

    print("Success.")


if __name__ == "__main__":
    main()

