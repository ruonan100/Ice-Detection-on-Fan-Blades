# Ice-Detection-on-Fan-Blades
1. data folder: 15000 sample dataset
2. tmp folder: trained model, Normalized parameters, Intermediate training result...

3. train_model_v0_3.ipynb: lstm、cnn、Autoencoding
4. train_model_v0_3.html: train_model_v0_3.ipynb

5. train_model_v0_3.py:
6. deployment.py: Trained model deployment code on the server. The program gets the latest data from the influxdb server at regular intervals, loads an lstm model as an example, predicts, and writes the results back to the database. 
