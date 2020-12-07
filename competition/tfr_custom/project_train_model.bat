set BASE="E:\coursera\Fall2020\cs410\CourseProject\competition"
set JSON_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition\json_data"
set TFRECORD_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition\tfr_custom\tfrecord_data"
set TUNED_MODEL_DIR="E:\coursera\Fall2020\cs410\CourseProject\competition\tfr_custom\tuned_listwise"

set CUDA_VISIBLE_DEVICES=0
::rmdir /s /q %TUNED_MODEL_DIR%

::cd "%BASE%\tfr_custom"

bazel build -c opt tensorflow_ranking/examples:tf_ranking_tfrecord_py_binary
"E:\coursera\Fall2020\cs410\CourseProject\competition\tfr_custom\bazel-bin\tensorflow_ranking\examples\tf_ranking_tfrecord_py_binary.exe" ^
    --train_path=%TFRECORD_DIR%\train.elwc.tfrecord ^
    --eval_path=%TFRECORD_DIR%\eval.elwc.tfrecord ^
    --vocab_path=%BASE%\datasets\scibert_vocab.txt ^
    --model_dir="%TUNED_MODEL_DIR%" ^
    --data_format=example_list_with_context ^
    --num_train_steps=200000 ^
    --learning_rate=.005 ^
    --dropout_rate=0.65 ^
    --list_size=500 ^
    --embedding_dim=1024 ^
    --loss=approx_ndcg_loss ^
    --listwise_inference ^
    --config=cuda
