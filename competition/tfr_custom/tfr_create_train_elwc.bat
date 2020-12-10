call tfr_set_vars.bat

python %WORKDIR%\tfr_convert_json_to_elwc.py ^
    --vocab_file %VOCAB_FILE% ^
    --sequence_length=%SEQ_LENGTH% ^
    --query_file=%JSON_DIR%/train_queries.json ^
    --qrel_file=%JSON_DIR%/train_qrels.json ^
    --doc_file=%JSON_DIR%/train_docs.json ^
    --query_key=%QUERY_TOKENS% ^
    --output_train_file=%WORKDIR%/tfrecord_data/train.elwc.tfrecord ^
    --output_eval_file=%WORKDIR%/tfrecord_data/eval.elwc.tfrecord ^
    --list_size=500 ^
    --do_lower_case
