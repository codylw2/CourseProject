@echo off
set WORKDIR=%~dp0
IF %WORKDIR:~-1%==\ SET WORKDIR=%WORKDIR:~0,-1%
set BASE=%WORKDIR%\..\..
set DATASET_DIR=%BASE%\competition\datasets
set JSON_DIR=%BASE%\competition\json_data
set TUNED_MODEL_DIR="%WORKDIR%\tuned_test"
set VOCAB_FILE=%DATASET_DIR%\scibert_vocab.txt

set CUDA_VISIBLE_DEVICES=0
set QUERY_TOKENS=narrative_tokens
set SEQ_LENGTH=1024
@echo on