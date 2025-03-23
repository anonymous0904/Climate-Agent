import validate_data
import pandas as pd


def write_metar_df_to_csv_file():
    metar_df = validate_data.get_validated_metar_data_frame()
    metar_df.to_csv('metars.csv', index=False, columns=metar_df.columns)
    print("Metars Data Frame added successfully.")


def read_metar_df_from_csv_file():
    return pd.read_csv('metars.csv')


print(read_metar_df_from_csv_file())
