@echo off
set WORKDIR=%~dp0
IF %WORKDIR:~-1%==\ SET WORKDIR=%WORKDIR:~0,-1%
set BASE=%WORKDIR%\..\..
set MODEL_DIR=%WORKDIR%\checkpoints\uncased_L-4_H-256_A-4_TF2
set DATASET_DIR=%BASE%\competition\datasets
set JSON_DIR=%BASE%\competition\json_data
set TUNED_MODEL_DIR="%WORKDIR%\tuned"
set VOCAB_FILE=%DATASET_DIR%\bert_vocab.txt

set CUDA_VISIBLE_DEVICES=0
set QUERY_TOKENS=question_tokens
set SEQ_LENGTH=512
@echo on