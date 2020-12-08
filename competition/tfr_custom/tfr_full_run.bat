call tfr_set_vars.bat

timeout /t 5400

:: create data to train with
call tfr_create_train_elwc.bat

:: train the model
call tfr_train_model.bat

:: generate predictions
call tfr_predict.bat
