call cm_set_vars.bat

python create_cranfield.py ^
    --run_type "test" ^
    --query_keys "query" ^
    --doc_keys "title:abstract:intro" ^
    --cranfield_dir "%WORKDIR%" ^
    --input_dir "%JSON_DIR%"
