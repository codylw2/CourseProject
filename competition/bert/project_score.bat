call project_set_vars.bat

python combine_scores.py ^
    --scores_input_dir %SCORE_DIR% ^
    --pred_key max ^
    --reload ^
    --scores_output_file test_scores_comb.json
