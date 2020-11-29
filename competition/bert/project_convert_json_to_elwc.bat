set BASE_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition"
set BERT_DIR="%BASE_DIR%\bert"
set MODEL_DIR="%BERT_DIR%\checkpoints\uncased_L-4_H-256_A-4_TF2"

python %BERT_DIR%/project_convert_json_to_elwc.py ^
    --vocab_file=%BERT_DIR%/vocab.txt ^
    --sequence_length=512 ^
    --query_file=%BASE_DIR%/train_queries.json ^
    --qrel_file=%BASE_DIR%/train_qrels.json ^
    --doc_file=%BASE_DIR%/train_docs.json ^
    --query_key=narrative_tokens ^
    --output_train_file=%BERT_DIR%/train.elwc.tfrecord ^
    --output_eval_file=%BERT_DIR%/eval.elwc.tfrecord ^
    --do_lower_case
