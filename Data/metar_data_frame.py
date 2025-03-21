from DatabaseHandler import DatabaseHandler
import pandas as pd

from Parser.metar_parser import metar_parser


# replace the code for each precipitation type with a unique number via label-encoding
def precipitation_for_observation(row):
    precipitation_codes = ["RA", "SN"]
    metar_phenomena_1 = row[['present_phenomena_1']]
    metar_phenomena_2 = row[['present_phenomena_2']]
    metar_phenomena_3 = row[['present_phenomena_3']]
    phenomenon_value = 0  # 0 - no present precipitation

    if any(precipitation_codes[1] in metar_phenomenon for metar_phenomenon in
           [metar_phenomena_1, metar_phenomena_2, metar_phenomena_3] if metar_phenomenon is not None):
        phenomenon_value = 2  # 2 - snow
    elif any(precipitation_codes[0] in metar_phenomenon for metar_phenomenon in
             [metar_phenomena_1, metar_phenomena_2, metar_phenomena_3] if metar_phenomenon is not None):
        phenomenon_value = 1  # 1 - rain

    return phenomenon_value


def present_fog_for_observation(row):
    fog_codes = ["BR", "FG"]
    metar_phenomena_1 = row[['present_phenomena_1']]
    metar_phenomena_2 = row[['present_phenomena_2']]
    metar_phenomena_3 = row[['present_phenomena_3']]
    present_fog = 0  # 0 for false

    if any(fog_code in metar_phenomenon for fog_code in fog_codes for metar_phenomenon in
           [metar_phenomena_1, metar_phenomena_2, metar_phenomena_3] if metar_phenomenon is not None):
        present_fog = 1

    return present_fog


# average cloud cover
def average_cloud_nebulosity_for_observation(row):
    cloud_nebulosity_mapping = {'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4}
    metar_cloud_nebulosity_1 = row.iloc[25]  # [['cloud_nebulosity_1']]
    metar_cloud_nebulosity_2 = row.iloc[28]  # [['cloud_nebulosity_2']]
    metar_cloud_nebulosity_3 = row.iloc[31]  # [['cloud_nebulosity_3']]
    nebulosity_sum = 0
    layers = 0

    if metar_cloud_nebulosity_1 is not None:
        nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_1]
        layers += 1
        if metar_cloud_nebulosity_2 is not None:
            nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_2]
            layers += 1
            if metar_cloud_nebulosity_3 is not None:
                nebulosity_sum += cloud_nebulosity_mapping[metar_cloud_nebulosity_3]
                layers += 1

    if layers > 0:
        return nebulosity_sum // layers
    else:
        return 0  # no clouds


def average_cloud_altitude_for_observation(row):
    metar_cloud_altitude_1 = row.iloc[26]
    metar_cloud_altitude_2 = row.iloc[29]
    metar_cloud_altitude_3 = row.iloc[32]
    altitude_sum = 0
    layers = 0
    if metar_cloud_altitude_1 is not None:
        altitude_sum += metar_cloud_altitude_1
        layers += 1
        if metar_cloud_altitude_2 is not None:
            altitude_sum += metar_cloud_altitude_2
            layers += 1
            if metar_cloud_altitude_3 is not None:
                altitude_sum += metar_cloud_altitude_3
                layers += 1

    if layers > 0:
        return altitude_sum // layers
    else:
        return 0


def get_metar_data_frame():
    metars_df = DatabaseHandler().get_metars_df()[5290::2]  # starting from the first day of 2022 and taking every hour
    metars_df['precipitation'] = metars_df.apply(precipitation_for_observation, axis=1)
    metars_df['present_fog'] = metars_df.apply(present_fog_for_observation, axis=1)
    metars_df['cloud_nebulosity'] = metars_df.apply(average_cloud_nebulosity_for_observation, axis=1)
    metars_df['cloud_altitude'] = metars_df.apply(average_cloud_altitude_for_observation, axis=1)
    metars_df = metars_df[
        ['observation_time', 'wind_direction', 'wind_speed', 'predominant_horizontal_visibility', 'precipitation',
         'present_fog', 'cloud_nebulosity', 'cloud_altitude', 'air_temperature', 'dew_point', 'air_pressure']]
    return metars_df


print(get_metar_data_frame()['cloud_altitude'])
