from DatabaseHandler import DatabaseHandler


# replace the code for each precipitation type with a unique number via label-encoding
def precipitation_for_observation(row):
    metar_phenomena_1 = str(row.iloc[22])
    metar_phenomena_2 = str(row.iloc[23])
    metar_phenomena_3 = str(row.iloc[24])
    phenomenon_value = 0  # 0 - no present precipitation

    if metar_phenomena_1 is None and metar_phenomena_2 is None and metar_phenomena_3 is None:
        return phenomenon_value

    if "RA" in metar_phenomena_1 or "RA" in metar_phenomena_2 or "RA" in metar_phenomena_3 or "SH" in metar_phenomena_1 or "SH" in metar_phenomena_2 or "SH" in metar_phenomena_3:
        phenomenon_value = 1  # 1 - rain
    elif "SN" in metar_phenomena_1 or "SN" in metar_phenomena_2 or "SN" in metar_phenomena_3:
        phenomenon_value = 2  # 2 - snow

    return phenomenon_value


def present_fog_for_observation(row):
    metar_phenomena_1 = str(row.iloc[22])
    metar_phenomena_2 = str(row.iloc[23])
    metar_phenomena_3 = str(row.iloc[24])
    present_fog = 0  # 0 for false

    if metar_phenomena_1 is None and metar_phenomena_2 is None and metar_phenomena_3 is None:
        return present_fog

    if "BR" in metar_phenomena_1 or "BR" in metar_phenomena_2 or "BR" in metar_phenomena_3 or "FG" in metar_phenomena_1 or "FG" in metar_phenomena_2 or "FG" in metar_phenomena_3:
        present_fog = 1

    return present_fog


# average cloud cover
def average_cloud_nebulosity_for_observation(row):
    cloud_nebulosity_mapping = {'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4}
    metar_cloud_nebulosity_1 = row.iloc[25]
    metar_cloud_nebulosity_2 = row.iloc[28]
    metar_cloud_nebulosity_3 = row.iloc[31]
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
    metar_cloud_altitude_1 = int(row.iloc[26])
    metar_cloud_altitude_2 = int(row.iloc[29])
    metar_cloud_altitude_3 = int(row.iloc[32])
    altitude_sum = 0
    layers = 0
    if metar_cloud_altitude_1 != 0:
        altitude_sum += metar_cloud_altitude_1
        layers += 1
        if metar_cloud_altitude_2 != 0:
            altitude_sum += metar_cloud_altitude_2
            layers += 1
            if metar_cloud_altitude_3 != 0:
                altitude_sum += metar_cloud_altitude_3
                layers += 1

    if layers > 0:
        return altitude_sum // layers
    else:
        return 0


def get_metar_data_frame():
    metars_df = DatabaseHandler().get_metars_df()[5290::2].sort_values(
        'id')  # starting from the first day of 2022 and taking every hour
    metars_df[['cloud_altitude_1', 'cloud_altitude_2', 'cloud_altitude_3']] = metars_df[
        ['cloud_altitude_1', 'cloud_altitude_2', 'cloud_altitude_3']].fillna(0).astype(int)
    metars_df['wind_direction'] = metars_df['wind_direction'].fillna(-1).astype(int)
    metars_df['wind_speed'] = metars_df['wind_speed'].astype(int)
    metars_df['precipitation'] = metars_df.apply(precipitation_for_observation, axis=1)
    metars_df['present_fog'] = metars_df.apply(present_fog_for_observation, axis=1)
    metars_df['cloud_nebulosity'] = metars_df.apply(average_cloud_nebulosity_for_observation, axis=1)
    metars_df['cloud_altitude'] = metars_df.apply(average_cloud_altitude_for_observation, axis=1)
    metars_df = metars_df[
        ['observation_time', 'wind_direction', 'wind_speed', 'predominant_horizontal_visibility', 'precipitation',
         'present_fog', 'cloud_nebulosity', 'cloud_altitude', 'air_temperature', 'dew_point', 'air_pressure']]
    return metars_df
