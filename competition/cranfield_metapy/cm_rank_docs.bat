call cm_set_vars.bat

python %WORKDIR%\search_eval.py ^
    --run_type "test" ^
    --ranker "bm25" ^
    --params "1.5;1.0;500" ^
    --dat_keys "title" ^
    --doc_weights "1.0" ^
    --cranfield_dir "%WORKDIR%" ^
    --predict_dir "%BASE%" ^
    --remove_idx
