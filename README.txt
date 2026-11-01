AEROSENSE — AQI PREDICTION SYSTEM 2.0

WHAT IS INCLUDED
- app.py: Streamlit web application with 3 sections
- data/demo_aqi_data.csv: synthetic demonstration data (NOT real monitoring data)
- report_template.docx: editable college report outline
- requirements.txt: Python dependencies

RUN ON WINDOWS
1. Extract the ZIP.
2. Open the inner AQI_Prediction_System_2_0 folder.
3. Click the File Explorer address bar, type cmd, press Enter.
4. Run: py -m pip install -r requirements.txt
5. Run: py -m streamlit run app.py
6. Open the local address shown in the terminal (usually http://localhost:8501).

FEATURES
- Home dashboard and charts
- AQI prediction using Random Forest, Decision Tree or Linear Regression
- Algorithm comparison with MAE, RMSE and R-squared
- Correlation analysis
- CSV upload and prediction export

IMPORTANT DATA NOTE
The bundled CSV is SYNTHETIC demo data generated for testing the app. It is not real AQI data and should not be cited as real observations. For final academic results, obtain a verified dataset from a credible source (for example, CPCB/official monitoring data or a documented research dataset), check units and AQI methodology, then upload a CSV with columns:
PM2.5, PM10, NO2, SO2, CO, O3, AQI

The uploaded data should have enough complete rows and consistent pollutant units. Model metrics depend on the dataset and are not guaranteed to be high. Do not claim the synthetic demo metrics as real-world accuracy.

TROUBLESHOOTING
- Keep the CMD window open while using the app.
- To stop the app, focus the CMD window and press Ctrl+C.
- To start it again later, repeat step 5 from the project folder.
