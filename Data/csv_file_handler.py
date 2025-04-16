# METHODS TO SAVE AND READ THE METAR DATA FRAME FROM THE CSV FILE

from validate_data import get_validated_metar_data_frame
import pandas as pd


# this method receives the validated data frame and saves it to the csv file
def write_metar_df_to_csv_file():
    metar_df = get_validated_metar_data_frame()
    metar_df.to_csv('metars.csv', index=False, columns=metar_df.columns)
    print("Metars Data Frame added successfully.")


def read_metar_df_from_csv_file():
    return pd.read_csv('metars.csv')
