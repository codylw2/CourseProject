call bert_set_vars.bat

python bert_predict.py ^
    --vocab_file %VOCAB_FILE% ^
    --sequence_length %SEQ_LENGTH% ^
    --query_file %JSON_DIR%/bert_test_queries.json ^
    --query_key %QUERY_TOKENS% ^
    --doc_file %JSON_DIR%/bert_test_docs.json ^
    --output_file %WORKDIR%\scores\test_scores.json ^
    --model_path %TUNED_MODEL_DIR% ^
    --docs_at_once 500 ^
    --rerank_file "%BASE%\predictions.txt" ^
    --do_lower_case
