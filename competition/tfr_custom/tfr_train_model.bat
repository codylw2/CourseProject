call tfr_set_vars.bat

rmdir /s /q %TUNED_MODEL_DIR%

cd %BASE%
bazel build -c opt tensorflow_ranking/examples:tf_ranking_tfrecord_py_binary

cd %WORKDIR%
"%BASE%\bazel-bin\tensorflow_ranking\examples\tf_ranking_tfrecord_py_binary.exe" ^
    --train_path="%WORKDIR%\tfrecord_data\train.elwc.tfrecord" ^
    --eval_path="%WORKDIR%\tfrecord_data\eval.elwc.tfrecord" ^
    --vocab_path=%VOCAB_FILE% ^
    --model_dir=%TUNED_MODEL_DIR% ^
    --data_format=example_list_with_context ^
    --num_train_steps=20000 ^
    --learning_rate=.005 ^
    --dropout_rate=0.65 ^
    --list_size=500 ^
    --embedding_dim=%SEQ_LENGTH% ^
    --loss=approx_ndcg_loss ^
    --listwise_inference ^
    --config=cuda
