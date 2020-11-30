call project_set_vars.bat

set CUDA_VISIBLE_DEVICES=0
rmdir /s /q %TUNED_MODEL_DIR%

bazel build -c opt tensorflow_ranking/extension/examples:tfrbert_example_py_binary
"%BERT_DIR%\bazel-bin\tensorflow_ranking\extension\examples\tfrbert_example_py_binary.exe" ^
    --train_input_pattern=%TFRECORD_DIR%/train.elwc.tfrecord ^
    --eval_input_pattern=%TFRECORD_DIR%/eval.elwc.tfrecord ^
    --bert_config_file=%MODEL_DIR%/bert_config.json ^
    --bert_init_ckpt=%MODEL_DIR%/bert_model.ckpt ^
    --bert_max_seq_length=512 ^
    --model_dir="%TUNED_MODEL_DIR%" ^
    --list_size=15 ^
    --loss=softmax_loss ^
    --train_batch_size=5 ^
    --eval_batch_size=5 ^
    --learning_rate=1e-5 ^
    --num_train_steps=20000 ^
    --num_eval_steps=100 ^
    --checkpoint_secs=100 ^
    --num_checkpoints=5 ^
    --config=cuda
