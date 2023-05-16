# download_data.py
# Purpose: to dowload the labeled water level training data

# Create data folder
mdkir data
cd data

# Download archives from Google Drive
gdown https://drive.google.com/uc?id=18_0d0l7tPNZiK3taAd_Klk1sKeG3-jRK
gdown https://drive.google.com/uc?id=14QvdFh3epLN1pgM8BRyXG94aG7etbe4t
gdown https://drive.google.com/uc?id=1Vg5wpFm6zhgfcunfMYweWC76ICjHWGaR
gdown https://drive.google.com/uc?id=13JzukjBZxIBo5WnP2_peNQnner_TkTlZ
gdown https://drive.google.com/uc?id=1g5lDFGYtw3D-0JUtHiYp3LKjnc8_sIIT

# Extract
unzip '*.zip'
