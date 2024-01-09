import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
PREPROCESSED_FILE_PATH = os.environ.get('PREPROCESSED_DATA_PATH').replace('/', os.path.sep)


def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f'Loaded dataset from {file_path}')
    return data

def save_data(data):
    data.to_csv(PREPROCESSED_FILE_PATH, index=False)
    print(f'Saved preprocessed data to {PREPROCESSED_FILE_PATH}')

def handle_missing_data(data, silent=False):
    none_categories = [
        'Zufahrtsweg', 'Kaminqualitaet', 'Zaunqualitaet', 'Poolqualitaet', 
        'Sondermerkmal', 'Garagenzustand', 
        'Garageninnenausbau', 'Garagenqualitaet', 'Garagentyp',
        'Kellerbelichtung', 'Kellerzustand', 'Kellerhoehe', 'Kellerbereich1',
        'Kellerbereich2'
    ]
    data[none_categories] = data[none_categories].fillna('None')

    mean_categories = ['Strassenlaenge']
    data[mean_categories] = data[mean_categories].fillna(data[mean_categories].mean())

    # Special cases
    data['Mauerwerktyp'] = data['Mauerwerktyp'].fillna('Kein')
    data = data.drop(['Garagenbaujahr'], axis=1)
    data = data.drop(['Id'], axis=1)

    majority_categories = [
        'Wohngebiet', 'Funktionalitaet', 'Versorgung', 'KuechenQualitaet',
        'Elektrik', 'Verkaufstyp', 
    ]
    data[majority_categories] = data[majority_categories].fillna(data[majority_categories].mode().iloc[0])

    zero_categories = [
        'KellerHalbbadezimmer', 'KellerVollbadezimmer', 'Garagenflaeche',
        'KellerbereichgroesseGes', 'KellerbereichgroesseNAu', 
        'Kellerbereichgroesse2', 'Kellerbereichgroesse1', 'Garagenautos',
        'Mauerwerkflaeche'
    ]
    data[zero_categories] = data[zero_categories].fillna(0)

    if not silent:
        print(
            'Amount of missing values after handling:', 
            data.isnull().sum().sum(),
            '✅︎' if data.isnull().sum().sum() == 0 else '❌'
        )
    return data

def encode_categories(data, silent=False): # One-hot encoding and label encoding
    one_hot_columns = [
        'Wohngebiet', 'Nachbarschaft', 'Bedingung1', 'Bedingung2',
        'Gebauedetyp', 'Wohnungsstil', 'Gelaendekontur', 'Grundstueckanordnung', 
        'Strassentyp', 'Zufahrtsweg', 'Fundament', 'Mauerwerktyp', 
        'Verkleidung1', 'Verkleidung2', 'Dachtyp', 'Dachmeterial', 'Heizung', 
        'Elektrik', 'Zaunqualitaet', 'Garagentyp', 'EinfahrtGepflastert', 
        'Sondermerkmal', 'Verkaufstyp', 'Verkaufsbedingung'
    ]
    data = pd.get_dummies(data, columns=one_hot_columns)

    versorgung_map = {
        'E': 0, 'EG': 1, 'EGW': 2, 'EGWA': 3
    }
    data['Versorgung'] = data['Versorgung'].map(versorgung_map)

    grundstuecksform_map = {'IR3': 0, 'UR2': 1, 'UR1': 2, 'Reg': 3}        
    data['Grundstuecksform'] = data['Grundstuecksform'].map(grundstuecksform_map)

    gelaendeneigung_map = {'San': 0, 'Mit': 1, 'Sta': 2}
    data['Gelaendeneigung'] = data['Gelaendeneigung'].map(gelaendeneigung_map)

    aussen_qualitaet_map = {'Sc': 0, 'Ar': 1, 'Du': 2, 'Gu': 3, 'Ag': 4}
    data['Aussenmaterialqualitaet'] = data['Aussenmaterialqualitaet'].map(aussen_qualitaet_map)
    data['Aussenmaterialzustand'] = data['Aussenmaterialzustand'].map(aussen_qualitaet_map)
    data['Heizungsqualitaet'] = data['Heizungsqualitaet'].map(aussen_qualitaet_map)
    data['KuechenQualitaet'] = data['KuechenQualitaet'].map(aussen_qualitaet_map)

    keller_qualitaet_map = {'None': 0, 'Sc': 1, 'Ar': 2, 'Ty': 3, 'Gu': 4, 'Ag': 5}
    data['Kellerhoehe'] = data['Kellerhoehe'].map(keller_qualitaet_map)
    data['Kellerzustand'] = data['Kellerzustand'].map(keller_qualitaet_map)

    kellerbelichtung_map = {'None': 0, 'Ke': 1, 'Mn': 2, 'Du': 3, 'Gu': 3}
    data['Kellerbelichtung'] = data['Kellerbelichtung'].map(kellerbelichtung_map)

    kellerbereich_map = {'None': 0, 'NAu': 1, 'SQ': 2, 'DAR': 3, 'UWR': 4, 'DWR': 5, 'GWR': 6}
    data['Kellerbereich1'] = data['Kellerbereich1'].map(kellerbereich_map)
    data['Kellerbereich2'] = data['Kellerbereich2'].map(kellerbereich_map)

    garagen_map = {'None': 0, 'Sc': 1, 'Ar': 2, 'Du': 3, 'Gu': 4, 'Ag': 5}
    data['Garagenqualitaet'] = data['Garagenqualitaet'].map(garagen_map)
    data['Garagenzustand'] = data['Garagenzustand'].map(garagen_map)
    data['Kaminqualitaet'] = data['Kaminqualitaet'].map(garagen_map)

    garagen_innen_map = {'None': 0, 'NAu': 1, 'GAu': 2, 'Aus': 3}
    data['Garageninnenausbau'] = data['Garageninnenausbau'].map(garagen_innen_map)

    pool_map = {'None': 0, 'Ar': 1, 'Du': 2, 'Gu': 3, 'Ag': 4}
    data['Poolqualitaet'] = data['Poolqualitaet'].map(pool_map)

    funktionalitaet_map = {'Ber':0, 'Sch': 1, 'Gro2': 2, 'Gro1': 3, 'Mit': 4, 'Ger2': 5, 'Ger1': 6, 'Typ': 7}
    data['Funktionalitaet'] = data['Funktionalitaet'].map(funktionalitaet_map)

    bool_map = {'J': True, 'N': False}
    data['Klimalanlage'] = data['Klimalanlage'].map(bool_map)
  
    if not silent:
        print('Amount of missing values after encoding:', data.isnull().sum().sum(), '✅︎' if data.isnull().sum().sum() == 0 else '❌')
        unique_cols = set([str(data[col].dtype) for col in data.columns])
        valid_cols = {'int64', 'bool', 'float64'}
        incorrect_cols = any(col not in valid_cols for col in unique_cols)
        print('Unique column types after encoding:', ', '.join(unique_cols), '✅︎' if not incorrect_cols else '❌')
    return data

def preprocess_data(data):
    print('Start preprocessing the data...')
    data = handle_missing_data(data)
    data = encode_categories(data)
    data.sort_index(axis=1, inplace=True)
    return data
