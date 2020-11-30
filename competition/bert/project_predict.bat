call project_set_vars.bat

set CUDA_VISIBLE_DEVICES=0

python project_predict.py ^
    --vocab_file %BERT_DIR%/vocab.txt ^
    --sequence_length 512 ^
    --query_file %JSON_DIR%/test_queries.json ^
    --query_key narrative_tokens ^
    --doc_file %JSON_DIR%/train_docs.json ^
    --output_file %SCORE_DIR%/test_scores.json ^
    --model_path %TUNED_MODEL_DIR%/export/best_model_by_loss ^
    --docs_at_once 550 ^
    --do_lower_case
