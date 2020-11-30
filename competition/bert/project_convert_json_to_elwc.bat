call project_set_vars.bat

python %BERT_DIR%/project_convert_json_to_elwc.py ^
    --vocab_file %BERT_DIR%/vocab.txt ^
    --sequence_length=512 ^
    --query_file=%JSON_DIR%/train_queries.json ^
    --qrel_file=%JSON_DIR%/train_qrels.json ^
    --doc_file=%JSON_DIR%/train_docs.json ^
    --query_key=narrative_tokens ^
    --output_train_file=%TFRECORD_DIR%/train.elwc.tfrecord ^
    --output_eval_file=%TFRECORD_DIR%/eval.elwc.tfrecord ^
    --do_lower_case
