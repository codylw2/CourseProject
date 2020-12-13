call cm_set_vars.bat

python %WORKDIR%\search_eval.py ^
    --run_type "test" ^
    --ranker "bm25" ^
    --params "2.0;0.9;4450" ^
    --dat_keys "title" ^
    --doc_weights "1.0" ^
    --cranfield_dir "%WORKDIR%" ^
    --predict_dir "%BASE%" ^
    --remove_idx
