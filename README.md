LSTM Time Series Forecasting – Jena Climate Project

I worked on this project to practice multivariate time series forecasting. I used the Jena Climate dataset because it has many weather features and enough data to train an LSTM model properly. My goal was to predict the temperature using previous readings of pressure, wind velocity, air density, and a few other values. The notebook includes everything from loading the data, preparing sequences, training the model, tuning some parameters, and generating SHAP explanations.

At first, the dataset was too big and the training took a long time. So I reduced the number of rows and also cut down the epochs during tuning. This helped a lot because Colab was slowing down. I used TimeSeriesSplit because random splits don’t make sense for time-based data. The tuning part tested different LSTM units and dropout values. I selected the best setup based on the lowest MAE across the folds.

After training the final model, I generated predictions on a test split. The error values looked reasonable and the model seemed to pick up the general patterns of the data. The SHAP part helped me see which features had more impact on the forecast. Pressure and air density showed more influence in most examples, while wind speed sometimes pushed the prediction slightly down. I added three SHAP local plots which show how different features affected specific test samples.

Overall, the project covers data prep, model building, validation, tuning, and explainability in one workflow. I kept the code simple so it can be followed easily. If someone wants to run it, they just need to place the dataset in the data/ folder and open the notebook. The outputs like the model file and charts are already saved in the outputs folder.

Folder Structure
TimeSeries_LSTM_Project/
│
├── README.md
│
├── notebooks/
│   └── LSTM_forecasting_explainability.ipynb
│
├── outputs/
│   ├── final_lstm.h5
│   ├── metrics_summary.csv
│   ├── cv_metrics.png
│   ├── loss_curve.png
│   ├── shap_summary.png
│   ├── shap_local_1.png
│   ├── shap_local_2.png
│   ├── shap_local_3.png
│   ├── shap_values.npy
│   └── scaler.save
│
└── data/
    └── .gitkeep

How to Run

Put jena_climate_2009_2016.csv inside data/

Open the notebook from the notebooks folder.

Run all cells in order.

Outputs will be saved automatically into the outputs/ folder.

Notes

I kept the epochs low in tuning to make sure it runs smoothly in Colab.

The project is mainly for learning, not for competition-level accuracy.

SHAP KernelExplainer was used because DeepExplainer had issues with TensorFlow 2.12.

End

If you need help reproducing the results or checking the code, all steps are inside the notebook