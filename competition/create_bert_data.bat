@echo off
set WORKDIR=%~dp0
IF %WORKDIR:~-1%==\ SET WORKDIR=%WORKDIR:~0,-1%
set BASE=%WORKDIR%\..
set DATASET_DIR=%BASE%\competition\datasets
set JSON_DIR=%BASE%\competition\json_data
set VOCAB_FILE=%DATASET_DIR%\bert_vocab.txt
@echo on

python create_bert_data.py ^
    --vocab_file %VOCAB_FILE% ^
    --run_type "train;test" ^
    --input_dir %DATASET_DIR% ^
    --output_dir %JSON_DIR% ^
    --output_prefix "bert" ^
    --tokenize
