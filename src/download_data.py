import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

dirs = [PROJECT_ROOT / "data" / "raw_data", PROJECT_ROOT / "data" / "preprocessed", 
        PROJECT_ROOT / "outputs" / "figures", PROJECT_ROOT / "models"]

for dir in dirs:
    dir.mkdir(parents=True, exist_ok=True)

urls = ['https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv'
        ,'https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-last-5-years.csv'
        ,'https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-casualty-last-5-years.csv']

names = ['collision', 'vehicle', 'casualty']

for i in range(3):
    response = requests.get(urls[i])
    path = PROJECT_ROOT / ('data/raw_data/' + names[i] + '_raw.csv')

    if response.status_code == 200:
        with open(path, 'wb') as file:
            file.write(response.content)
        print('Success')
    else:
        print('Fail')