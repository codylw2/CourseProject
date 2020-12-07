set BASE="E:\coursera\Fall2020\cs410\CourseProject\competition"
set DATA_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition\datasets"
set JSON_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition\json_data"
set SCORE_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition\tfr_custom\scores"
set TFRECORD_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition\tfr_custom\tfrecord_data"
set TUNED_MODEL_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition\tfr_custom\tuned_listwise\export\saved_model_exporter\1607360060"

set CUDA_VISIBLE_DEVICES=0

python tfr_predict.py ^
    --vocab_file %DATA_DIR%/scibert_vocab.txt ^
    --sequence_length 1024 ^
    --query_file %JSON_DIR%/test_queries.json ^
    --query_key question_tokens ^
    --doc_file %JSON_DIR%/test_docs.json ^
    --output_file %SCORE_DIR%/test_scores.json ^
    --model_path %TUNED_MODEL_DIR% ^
    --docs_at_once 500 ^
    --rerank_file "E:\coursera\Fall2020\cs410\CourseProject\predictions.txt" ^
    --do_lower_case
