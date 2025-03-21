# ANALIZA STATISTICA DE BAZA A ERORILOR DE PROGNOZA


# CALCULUL ERORILOR DINTRE PROGNOZA SI OBSERVATIE. O DIFERENTA POZITIVA INDICA SUPRAESTIMAREA VALORII VARIABILEI,
# UNA NEGATIVA INDICA SUBESTIMAREA

# variabile de baza: predominant_horizontal_visibility, wind_direction(can be null), wind_speed,
# wind_variability(true/false), present_phenomena 1, 2 and 3 (can be null), cloud_nebulosity(can be null),
# cloud_altitude(can be null), presence of cumulonimbus(true/false), vertical visibility(most of them are null)
# date in plus: sezon, ora din zi, air_temperature, dew_point, air_pressure
# fenomene meteo:
# ** CARE REDUC VIZIBILITATEA: BR(aer cetos),FG(ceata),TS(oraj-fulgere si tunete), VC(in vecintatate-apare inaintea fenomenului)
# ** PRECIPITATII: RA(ploaie),SH(averse),DZ(burnita)

# calculate the difference between the forecasted and observed values
def calculate_taf_wind_direction_error_for_row(row):
    if row.iloc[53] % 360 is None:  # if there is variation in the wind direction
        # (the correct value should be between 60 and 180 degrees)
        if row.iloc[5] % 360 is not None and row.iloc[5] % 360 < row.iloc[57] % 360:
            return row.iloc[5] % 360 - row.iloc[57] % 360
        elif row.iloc[5] is not None and row.iloc[5] > row.iloc[58]:
            return row.iloc[5] % 360 - row.iloc[58] % 360

    if row.iloc[5] % 360 is not None:
        return row.iloc[5] % 360 - row.iloc[53] % 360
    else:
        return None  # no data for the variable


def calculate_taf_prob_wind_direction_error_for_row(row):
    if row.iloc[53] % 360 is None:  # if there is variation in the wind direction
        # (the correct value should be between 60 and 180 degrees)
        if row.iloc[30] % 360 is not None and row.iloc[30] % 360 < row.iloc[57] % 360:
            return row.iloc[30] % 360 - row.iloc[57] % 360
        elif row.iloc[30] % 360 is not None and row.iloc[30] % 360 > row.iloc[58] % 360:
            return row.iloc[30] % 360 - row.iloc[58] % 360

    if row.iloc[30] % 360 is not None:
        return row.iloc[30] % 360 - row.iloc[53] % 360
    else:
        return None  # no data for the variable


# def calculate_wind_direction_error_for_row(row):
#     # if there is variation in the wind direction (the correct value should be between 60 and 180 degrees)
#     if row['wind_direction'][2] is None:
#
#         # first, check whether the probable phenomena took place or not (if the difference between the metar and taf
#         # wind direction is smaller than 30 degrees we consider it valid)
#         if row['wind_direction'][1] is not None and row['wind_direction'][1] < row['lower_direction_variation'] and (
#                 row['lower_direction_variation'] - row['wind_direction'][1]) <= 30:
#             return row['wind_direction'][1] - row['lower_direction_variation']
#
#         elif row['wind_direction'][1] is not None and row['wind_direction'][1] > row['upper_direction_variation'] and (
#                 row['upper_direction_variation'] - row['wind_direction'][1]) <= 30:
#             return row['wind_direction'][1] - row['upper_direction_variation']
#
#         elif row['wind_direction'][1] is not None and row['lower_direction_variation'] <= row['wind_direction'][1] <= \
#                 row['upper_direction_variation']:
#             return 0
#
#         # then check the usual taf forecast
#         elif row['wind_direction'][0] is not None and row['wind_direction'][0] < row['lower_direction_variation']:
#             return row['wind_direction'][0] - row['lower_direction_variation']
#         elif row['wind_direction'][0] is not None and row['wind_direction'][0] > row['upper_direction_variation']:
#             return row['wind_direction'][0] - row['upper_direction_variation']
#         elif row['wind_direction'][0] is not None and row['lower_direction_variation'] <= row['wind_direction'][0] <= \
#                 row['upper_direction_variation']:
#             return 0
#
#     if row['wind_direction'][1] is not None and abs(row['wind_direction'][1] - row['wind_direction'][
#         2]) <= 30:  # first, check whether the probable phenomena took place or not
#         return row['wind_direction'][1] - row['wind_direction'][2]
#     elif row['wind_direction'][0] is not None:
#         return row['wind_direction'][0] - row['wind_direction'][2]
#     else:
#         return None  # no data for the variable

# how to use the method: creates a new column in the data frame and applies the method to each row
# data_frame['wind_direction_error'] = data_frame.apply(calculate_wind_direction_error_for_row,axis=1)


def calculate_taf_wind_speed_error_for_row(row):
    if row.iloc[54] is None:
        return None
    if row.iloc[6] is not None:
        return row.iloc[6] - row.iloc[54]
    else:
        return None


def calculate_taf_prob_wind_speed_error_for_row(row):
    if row.iloc[54] is None:
        return None
    if row.iloc[31] is not None:
        return row.iloc[31] - row.iloc[54]
    else:
        return None


# direction variability
def calculate_taf_wind_variability_error_for_row(row):
    # forecasted wind variability that did not occur
    if row.iloc[7] == True and row.iloc[55] == False:
        return 1
    # did not forecast wind variability but it did occur
    elif row.iloc[7] == False and row.iloc[55] == True:
        return -1
    # correct forecast
    else:
        return 0


def calculate_taf_prob_wind_variability_error_for_row(row):
    # forecasted wind variability that did not occur
    if row.iloc[32] == True and row.iloc[55] == False:
        return 1
    # did not forecast wind variability but it did occur
    elif row.iloc[32] == False and row.iloc[55] == True:
        return -1
    # correct forecast
    else:
        return 0


def calculate_taf_gust_speed_error_for_row(row):
    if row.iloc[8] is not None and row.iloc[56] is not None:
        return row.iloc[8] - row.iloc[56]
    else:
        return None


def calculate_taf_prob_gust_speed_error_for_row(row):
    if row.iloc[33] is not None and row.iloc[56] is not None:
        return row.iloc[33] - row.iloc[56]
    else:
        return None


def calculate_taf_horizontal_visibility_error_for_row(row):
    return row.iloc[10] - row.iloc[60]


def calculate_taf_prob_horizontal_visibility_error_for_row(row):
    if row.iloc[35] is not None:
        return row.iloc[35] - row.iloc[60]


def calculate_taf_rainfall_error_for_row(row):
    rainfall_codes = ["DZ", "RA", "SN", "SG", "IC", "PL", "GR", "GS", "UP", "SH"]
    taf_phenomena_1 = row.iloc[11]
    taf_phenomena_2 = row.iloc[12]
    taf_phenomena_3 = row.iloc[13]
    metar_phenomena_1 = row.iloc[71]
    metar_phenomena_2 = row.iloc[72]
    metar_phenomena_3 = row.iloc[73]

    if any(rainfall_code in taf_phenomenon for rainfall_code in rainfall_codes for taf_phenomenon in
           [taf_phenomena_1, taf_phenomena_2, taf_phenomena_3] if taf_phenomenon is not None):
        rainfall_forecasted = True
    else:
        rainfall_forecasted = False
    if any(rainfall_code in metar_phenomenon for rainfall_code in rainfall_codes for metar_phenomenon in
           [metar_phenomena_1, metar_phenomena_2, metar_phenomena_3] if metar_phenomenon is not None):
        rainfall_occurred = True
    else:
        rainfall_occurred = False

    if rainfall_forecasted and not rainfall_occurred:
        return 1
    if not rainfall_forecasted and rainfall_occurred:
        return -1
    else:
        return 0


def calculate_taf_prob_rainfall_error_for_row(row):
    rainfall_codes = ["DZ", "RA", "SN", "SG", "IC", "PL", "GR", "GS", "UP", "SH"]
    taf_phenomena_1 = row.iloc[36]
    taf_phenomena_2 = row.iloc[37]
    taf_phenomena_3 = row.iloc[38]
    metar_phenomena_1 = row.iloc[71]
    metar_phenomena_2 = row.iloc[72]
    metar_phenomena_3 = row.iloc[73]

    if any(rainfall_code in taf_phenomenon for rainfall_code in rainfall_codes for taf_phenomenon in
           [taf_phenomena_1, taf_phenomena_2, taf_phenomena_3] if taf_phenomenon is not None):
        rainfall_forecasted = True
    else:
        rainfall_forecasted = False
    if any(rainfall_code in metar_phenomenon for rainfall_code in rainfall_codes for metar_phenomenon in
           [metar_phenomena_1, metar_phenomena_2, metar_phenomena_3] if metar_phenomenon is not None):
        rainfall_occurred = True
    else:
        rainfall_occurred = False

    if rainfall_forecasted and not rainfall_occurred:
        return 1
    if not rainfall_forecasted and rainfall_occurred:
        return -1
    else:
        return 0


def calculate_taf_fog_error_for_row(row):
    fog_codes = ["BR", "FG"]
    taf_phenomena_1 = row.iloc[11]
    taf_phenomena_2 = row.iloc[12]
    taf_phenomena_3 = row.iloc[13]
    metar_phenomena_1 = row.iloc[71]
    metar_phenomena_2 = row.iloc[72]
    metar_phenomena_3 = row.iloc[73]

    if any(fog_code in taf_phenomenon for fog_code in fog_codes for taf_phenomenon in
           [taf_phenomena_1, taf_phenomena_2, taf_phenomena_3] if taf_phenomenon is not None):
        fog_forecasted = True
    else:
        fog_forecasted = False
    if any(fog_code in metar_phenomenon for fog_code in fog_codes for metar_phenomenon in
           [metar_phenomena_1, metar_phenomena_2, metar_phenomena_3] if metar_phenomenon is not None):
        fog_occurred = True
    else:
        fog_occurred = False

    if fog_forecasted and not fog_occurred:
        return 1
    if not fog_forecasted and fog_occurred:
        return -1
    else:
        return 0


def calculate_taf_prob_fog_error_for_row(row):
    fog_codes = ["BR", "FG"]
    taf_phenomena_1 = row.iloc[36]
    taf_phenomena_2 = row.iloc[37]
    taf_phenomena_3 = row.iloc[38]
    metar_phenomena_1 = row.iloc[71]
    metar_phenomena_2 = row.iloc[72]
    metar_phenomena_3 = row.iloc[73]

    if any(fog_code in taf_phenomenon for fog_code in fog_codes for taf_phenomenon in
           [taf_phenomena_1, taf_phenomena_2, taf_phenomena_3] if taf_phenomenon is not None):
        fog_forecasted = True
    else:
        fog_forecasted = False
    if any(fog_code in metar_phenomenon for fog_code in fog_codes for metar_phenomenon in
           [metar_phenomena_1, metar_phenomena_2, metar_phenomena_3] if metar_phenomenon is not None):
        fog_occurred = True
    else:
        fog_occurred = False

    if fog_forecasted and not fog_occurred:
        return 1
    if not fog_forecasted and fog_occurred:
        return -1
    else:
        return 0


# calculate the sum of the cloud cover for all the layers
def calculate_taf_cloud_nebulosity_error_for_row(row):
    cloud_nebulosity_mapping = {'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4}
    taf_cloud_nebulosity_1 = row.iloc[14]
    taf_cloud_nebulosity_2 = row.iloc[17]
    taf_cloud_nebulosity_3 = row.iloc[20]
    metar_cloud_nebulosity_1 = row.iloc[74]
    metar_cloud_nebulosity_2 = row.iloc[77]
    metar_cloud_nebulosity_3 = row.iloc[80]

    total_taf_nebulosity_sum = 0
    if taf_cloud_nebulosity_1 is not None:
        total_taf_nebulosity_sum += cloud_nebulosity_mapping[taf_cloud_nebulosity_1]
        if taf_cloud_nebulosity_2 is not None:
            total_taf_nebulosity_sum += cloud_nebulosity_mapping[taf_cloud_nebulosity_2]
            if taf_cloud_nebulosity_3 is not None:
                total_taf_nebulosity_sum += cloud_nebulosity_mapping[taf_cloud_nebulosity_3]

    total_metar_nebulosity_sum = 0
    if metar_cloud_nebulosity_1 is not None:
        total_metar_nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_1]
        if metar_cloud_nebulosity_2 is not None:
            total_metar_nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_2]
            if metar_cloud_nebulosity_3 is not None:
                total_metar_nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_3]

    return total_taf_nebulosity_sum - total_metar_nebulosity_sum


def calculate_taf_prob_cloud_nebulosity_error_for_row(row):
    cloud_nebulosity_mapping = {'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4}
    taf_cloud_nebulosity_1 = row.iloc[39]
    taf_cloud_nebulosity_2 = row.iloc[42]
    taf_cloud_nebulosity_3 = row.iloc[45]
    metar_cloud_nebulosity_1 = row.iloc[74]
    metar_cloud_nebulosity_2 = row.iloc[77]
    metar_cloud_nebulosity_3 = row.iloc[80]

    total_taf_prob_nebulosity_sum = 0
    if taf_cloud_nebulosity_1 is not None:
        total_taf_prob_nebulosity_sum += cloud_nebulosity_mapping[taf_cloud_nebulosity_1]
        if taf_cloud_nebulosity_2 is not None:
            total_taf_prob_nebulosity_sum += cloud_nebulosity_mapping[taf_cloud_nebulosity_2]
            if taf_cloud_nebulosity_3 is not None:
                total_taf_prob_nebulosity_sum += cloud_nebulosity_mapping[taf_cloud_nebulosity_3]

    total_metar_nebulosity_sum = 0
    if metar_cloud_nebulosity_1 is not None:
        total_metar_nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_1]
        if metar_cloud_nebulosity_2 is not None:
            total_metar_nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_2]
            if metar_cloud_nebulosity_3 is not None:
                total_metar_nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_3]

    return total_taf_prob_nebulosity_sum - total_metar_nebulosity_sum


# calculate the difference between the lowest cloud altitudes
def calculate_taf_lowest_cloud_altitude_error_for_row(row):
    taf_cloud_altitude_1 = row.iloc[15]
    taf_cloud_altitude_2 = row.iloc[18]
    taf_cloud_altitude_3 = row.iloc[21]
    metar_cloud_altitude_1 = row.iloc[75]
    metar_cloud_altitude_2 = row.iloc[78]
    metar_cloud_altitude_3 = row.iloc[81]

    min_taf_altitude = None
    if taf_cloud_altitude_1 is not None:
        min_taf_altitude = taf_cloud_altitude_1
    if taf_cloud_altitude_2 is not None and taf_cloud_altitude_2 < min_taf_altitude:
        min_taf_altitude = taf_cloud_altitude_2
    if taf_cloud_altitude_3 is not None and taf_cloud_altitude_3 < min_taf_altitude:
        min_taf_altitude = taf_cloud_altitude_3

    min_metar_altitude = None
    if metar_cloud_altitude_1 is not None:
        min_metar_altitude = metar_cloud_altitude_1
    if metar_cloud_altitude_2 is not None and metar_cloud_altitude_2 < min_metar_altitude:
        min_metar_altitude = metar_cloud_altitude_2
    if metar_cloud_altitude_3 is not None and metar_cloud_altitude_3 < min_metar_altitude:
        min_metar_altitude = metar_cloud_altitude_3

    if min_taf_altitude is not None and min_metar_altitude is not None:
        return min_taf_altitude - min_metar_altitude
    else:
        return None


def calculate_taf_prob_lowest_cloud_altitude_error_for_row(row):
    taf_cloud_altitude_1 = row.iloc[40]
    taf_cloud_altitude_2 = row.iloc[43]
    taf_cloud_altitude_3 = row.iloc[46]
    metar_cloud_altitude_1 = row.iloc[75]
    metar_cloud_altitude_2 = row.iloc[78]
    metar_cloud_altitude_3 = row.iloc[81]

    min_taf_altitude = None
    if taf_cloud_altitude_1 is not None:
        min_taf_altitude = taf_cloud_altitude_1
    if taf_cloud_altitude_2 is not None and taf_cloud_altitude_2 < min_taf_altitude:
        min_taf_altitude = taf_cloud_altitude_2
    if taf_cloud_altitude_3 is not None and taf_cloud_altitude_3 < min_taf_altitude:
        min_taf_altitude = taf_cloud_altitude_3

    min_metar_altitude = None
    if metar_cloud_altitude_1 is not None:
        min_metar_altitude = metar_cloud_altitude_1
    if metar_cloud_altitude_2 is not None and metar_cloud_altitude_2 < min_metar_altitude:
        min_metar_altitude = metar_cloud_altitude_2
    if metar_cloud_altitude_3 is not None and metar_cloud_altitude_3 < min_metar_altitude:
        min_metar_altitude = metar_cloud_altitude_3

    if min_taf_altitude is not None and min_metar_altitude is not None:
        return min_taf_altitude - min_metar_altitude
    else:
        return None


def calculate_taf_cloud_layers_error_for_row(row):
    taf_cloud_layer_1 = row.iloc[14]
    taf_cloud_layer_2 = row.iloc[17]
    taf_cloud_layer_3 = row.iloc[20]
    metar_cloud_layer_1 = row.iloc[74]
    metar_cloud_layer_2 = row.iloc[77]
    metar_cloud_layer_3 = row.iloc[80]

    taf_layer_count = 0
    metar_layer_count = 0
    if taf_cloud_layer_1 is not None:
        taf_layer_count += 1
    if taf_cloud_layer_2 is not None:
        taf_layer_count += 1
    if taf_cloud_layer_3 is not None:
        taf_layer_count += 1

    if metar_cloud_layer_1 is not None:
        metar_layer_count += 1
    if metar_cloud_layer_2 is not None:
        metar_layer_count += 1
    if metar_cloud_layer_3 is not None:
        metar_layer_count += 1

    return taf_layer_count - metar_layer_count


def calculate_taf_prob_cloud_layers_error_for_row(row):
    taf_cloud_layer_1 = row.iloc[40]
    taf_cloud_layer_2 = row.iloc[43]
    taf_cloud_layer_3 = row.iloc[46]
    metar_cloud_layer_1 = row.iloc[74]
    metar_cloud_layer_2 = row.iloc[77]
    metar_cloud_layer_3 = row.iloc[80]

    taf_layer_count = 0
    metar_layer_count = 0
    if taf_cloud_layer_1 is not None:
        taf_layer_count += 1
    if taf_cloud_layer_2 is not None:
        taf_layer_count += 1
    if taf_cloud_layer_3 is not None:
        taf_layer_count += 1

    if metar_cloud_layer_1 is not None:
        metar_layer_count += 1
    if metar_cloud_layer_2 is not None:
        metar_layer_count += 1
    if metar_cloud_layer_3 is not None:
        metar_layer_count += 1

    return taf_layer_count - metar_layer_count


def get_month_of_the_row(row):
    date = row['observation_time']
    return date.month


def get_season_of_the_row(row):
    date = row['observation_time']
    month = date.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"


def get_time_of_day_for_row(row):
    hour = row['observation_time'].hour
    if 6 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 15:
        return 'noon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'
