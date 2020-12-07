set BERT_DIR="E:\coursera\Fall2020\cs410\rankers_testing\ranking_2\checkpoints\scibert_scivocab_uncased"
set JSON_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition"
set TFRECORD_DIR="E:\coursera\Fall2020\cs410\rankers_testing\ranking_2\tfrecord_data"

python project_convert_json_to_elwc.py ^
    --vocab_file %BERT_DIR%/vocab.txt ^
    --sequence_length=1024 ^
    --query_file=%JSON_DIR%/train_queries.json ^
    --qrel_file=%JSON_DIR%/train_qrels.json ^
    --doc_file=%JSON_DIR%/train_docs.json ^
    --query_key=question_tokens ^
    --output_train_file=%TFRECORD_DIR%/train.elwc.tfrecord ^
    --output_eval_file=%TFRECORD_DIR%/eval.elwc.tfrecord ^
    --list_size=500 ^
    --do_lower_case
