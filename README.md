# Uzbekistan River Discharge Anomaly Detection

Comparative evaluation of LSTM, Conv-LSTM and Wavelet-LSTM 
models for hydrological anomaly detection in Uzbekistan river basins.

## Study Stations
- Chatkal (Chirchiq basin, 1932-2016)
- Karadarya (Syr Darya basin, 1914-2016)  
- Qashqadarya (Qashqadarya basin, 1926-2016)
- Tupalang (Surkhandarya basin, 1993-2016)

## Data Sources
- CA-discharge dataset (Marti et al., 2023): 
  https://doi.org/10.5281/zenodo.8147591
- ERA5-Land reanalysis (ECMWF): 
  https://doi.org/10.24381/cds.68d2bb30

## Requirements
- Python 3.12
- TensorFlow 2.x
- pandas, NumPy, SciPy, scikit-learn
- matplotlib

## Usage
1. Download CA-discharge.gpkg from Zenodo
2. Download ERA5-Land data from Copernicus CDS
3. Run convlstm_hydrology.py in Google Colab or local environment

## Citation
Nasridinov, R.B. (2026) Comparative Evaluation of LSTM, Conv-LSTM 
and Wavelet-LSTM Models for Hydrological Anomaly Detection in 
Uzbekistan River Basins. International Journal of Hydrology 
Science and Technology (under review).

## Author
Rustamjon Nasridinov
Tashkent University of Information Technologies (TUIT)
r.nasridinov@tuit.uz
