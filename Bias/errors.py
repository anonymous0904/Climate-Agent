import DatabaseHandler


# ANALIZA STATISTICA DE BAZA

# CALCULUL ERORILOR DINTRE PROGNOZA SI OBSERVATIE. O DIFERENTA POZITIVA INDICA SUPRAESTIMAREA VALORII VARIABILEI,
# UNA NEGATIVA INDICA SUBESTIMAREA

# variabile de baza: predominant_horizontal_visibility, wind_direction(can be null), wind_speed,
# wind_variability(true/false), cavok, present_phenomena 1, 2 and 3 (can be null), cloud_nebulosity(can be null),
# cloud_altitude(can be null), presence of cumulonimbus(true/false), vertical visibility(most of them are null)
# date in plus: sezon, ora din zi, air_temperature, dew_point, air_pressure
# fenomene meteo:
# ** CARE REDUC VIZIBILITATEA: BR(aer cetos),FG(ceata),TS(oraj-fulgere si tunete), VC(in vecintatate-apare inaintea fenomenului)
# ** PRECIPITATII: RA(ploaie),SH(averse),DZ(burnita)

# calculate the difference between the forecasted and observed values
def calculate_wind_direction_error_for_row(row):
    # if there is variation in the wind direction (the correct value should be between 60 and 180 degrees)
    if row['wind_direction'][2] is None:

        # first, check whether there is probable phenomena or not
        if row['wind_direction'][1] is not None and row['wind_direction'][1] < 60:
            return row['wind_direction'][1] - 60
        elif row['wind_direction'][1] is not None and row['wind_direction'][1] > 180:
            return row['wind_direction'][1] - 180

        # then check the usual taf forecast
        elif row['wind_direction'][0] is not None and row['wind_direction'][0] < 60:
            return row['wind_direction'][0] - 60
        elif row['wind_direction'][0] is not None and row['wind_direction'][0] > 180:
            return row['wind_direction'][0] - 180

    # if there are values forecasted with a certain probability of occurrence
    if row['wind_direction'][1] is not None:
        return row['wind_direction'][1] - row['wind_direction'][2]
    elif row['wind_direction'][0] is not None:
        return row['wind_direction'][0] - row['wind_direction'][2]
    else:
        return None  # no data for tha variable

    # how to use the method: creates a new column in the data frame and applies the method to each row
    # data_frame['wind_direction_error'] = data_frame.apply(calculate_wind_direction_error_for_row,axis=1)


def calculate_wind_speed_error_for_row(row):
    if row['wind_speed'][1] is not None:  # if there is probable phenomena in the forecast
        return row['wind_speed'][1] - row['wind_speed'][2]
    elif row['wind_speed'][0] is not None:
        return row['wind_speed'][0] - row['wind_speed'][2]
    else:
        return None


def calculate_wind_variability_error_for_row(row):
    # forecasted wind variability that did not occur
    if (row['wind_variability'][0] == True or row['wind_variability'][1] == True) and row['wind_variability'][
        2] == False:
        return 1
    # did not forecast wind variability but it did occur
    elif (row['wind_variability'][0] == False and row['wind_variability'][1] == False) and row['wind_variability'][
        2] == True:
        return -1
    # correct forecast
    else:
        return 0


def calculate_gust_speed_error_for_row(row):
    if row['gust_speed'][1] is not None:  # if there is probable phenomena in the forecast
        return row['gust_speed'][1] - row['gust_speed'][2]
    elif row['gust_speed'][0] is not None:
        return row['gust_speed'][0] - row['gust_speed'][2]
    else:
        return None


def horizontal_visibility_error_for_row(row):
    if row['horizontal_visibility'][1] is not None:  # if there is probable phenomena in the forecast
        return row['horizontal_visibility'][1] - row['predominant_horizontal_visibility']
    elif row['horizontal_visibility'][0] is not None:
        return row['horizontal_visibility'][0] - row['predominant_horizontal_visibility']
    else:
        return None


class Errors:
    def __init__(self):
        self.tafs_metars_df = DatabaseHandler.DatabaseHandler().get_taf_probs_metars_df()
