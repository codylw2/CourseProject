@echo off
set WORKDIR=%~dp0
IF %WORKDIR:~-1%==\ SET WORKDIR=%WORKDIR:~0,-1%
set BASE=%WORKDIR%\..\..
set DATASET_DIR=%BASE%\competition\datasets
set JSON_DIR=%BASE%\competition\json_data

set CUDA_VISIBLE_DEVICES=0
@echo on