call cm_set_vars.bat

python search_eval.py ^
    --run_type "test" ^
    --ranker "bm25+" ^
    --params "1.0;0.38;1.0" ^
    --dat_keys "title;abstract;text" ^
    --doc_weights "0.8;0.5;0.0" ^
    --cranfield_dir "%WORKDIR%" ^
    --predict_dir "%BASE%" ^
    --remove_idx
