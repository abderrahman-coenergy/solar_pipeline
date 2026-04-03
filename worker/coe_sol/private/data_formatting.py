import json
import pandas as pd

def extract_pyrano_data(dataJSON: dict) -> pd.DataFrame:
    '''
    dataJSON must contain the following fields:
    {
        "pyranos-setup": {
            "latitude": float,
            "longitude": float,
            "elevation": float,
        },
        "values": [
            {
                "time": str, # format: "YYYY-MM-DD HH:MM:SS",
                ... # other instruments, ignored
                "pyrano-origin": float, # in W/m^2
                "pyrano-fit-i": { # i can be 1, 2, 3, 4, 5, ..., they are the fit pyranometres. A max of 5 fit pyranometers is assumed.
                    "azimuth": float, # in degrees, 0 is north
                    "tilt": float, # in degrees, 0 is horizontal
                    "value": float # in W/m^2
                },
                "pyrano-dest-i": { # i can be 1, 2, 3, 4, 5, ..., they are the destination pyranometres
                    "azimuth": float, # in degrees, 0 is north
                    "tilt": float,
                    "value": float, # leave empty, it will be computed by this model
                }
            }
        ]
    }

    Returns a pandas DataFrame with the following columns:
    - time: datetime
    - pyrano-origin: float
    - pyrano-fit-i_azimuth: float
    - pyrano-fit-i_tilt: float
    - pyrano-fit-i_value: float
    - pyrano-dest-i_azimuth: float
    - pyrano-dest-i_tilt: float
    - pyrano-dest-i_value: float

    columns for pyrano-dest-i_value will be empty in the JSON so it
    will be filled with zeros in the DataFrame.

    Example usage:

    dataJSON = { ... }  # Your JSON data here
    df = extract_pyrano_data(dataJSON)
    print(df)

    '''
    records = []
    for entry in dataJSON['values']:
        record = {
            'time': entry['time'],
            'pyrano-origin': entry['pyrano-origin']
        }
        for i in range(1, 6):
            fit_key = f'pyrano-fit-{i}'
            dest_key = f'pyrano-dest-{i}'
            if fit_key in entry:
                record[f'{fit_key}_azimuth'] = entry[fit_key]['azimuth']
                record[f'{fit_key}_tilt'] = entry[fit_key]['tilt']
                record[f'{fit_key}_value'] = entry[fit_key]['value']
            if dest_key in entry:
                record[f'{dest_key}_azimuth'] = entry[dest_key]['azimuth']
                record[f'{dest_key}_tilt'] = entry[dest_key]['tilt']
                record[f'{dest_key}_value'] = entry[dest_key].get('value', 0.0)
                record[f'{dest_key}_value'] = entry[dest_key].get('value', 0.0)
        records.append(record)

    df = pd.DataFrame(records)
    return df
