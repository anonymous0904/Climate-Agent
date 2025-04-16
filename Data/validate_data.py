# THIS FILE RECEIVES THE TRANSFORMED METAR DATA FRAME FROM metar_data_frame AND VALIDATES
# THE VALUES WITHIN, AFTER THAT IT SEND THE VALIDATED DATA FRAME TO THE CSV FILE HANDLER

import numpy as np
import metar_data_frame


# calculate the circular mean of wind directions
def circular_mean_wind_direction(direction_angles, w=None):
    if w is None:
        w = np.ones(len(direction_angles))
    sin_sum = np.sum(w * np.sin(np.radians(direction_angles)))
    cos_sum = np.sum(w * np.cos(np.radians(direction_angles)))
    mean = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
    return int(mean)


def validate_wind_direction(data_frame):
    missing_indices = data_frame[data_frame['wind_direction'] == -1].index

    if missing_indices.empty:
        return data_frame

    wind_direction_to_validate = data_frame['wind_direction'].copy()
    for i in missing_indices:
        valid_before = data_frame['wind_direction'].loc[:i][data_frame['wind_direction'].loc[:i] != -1].index.tolist()
        valid_after = data_frame['wind_direction'].loc[i:][data_frame['wind_direction'].loc[i:] != -1].index.tolist()

        if not valid_before or not valid_after:
            wind_direction_to_validate.loc[i] = -1
            continue

        idx_before = valid_before[-1]
        idx_after = valid_after[0]

        dist_before = i - idx_before
        dist_after = idx_after - i
        total_dist = dist_before + dist_after

        weight_before = dist_after / total_dist
        weight_after = dist_before / total_dist

        direction_angles = [data_frame['wind_direction'][idx_before], data_frame['wind_direction'][idx_after]]
        weights = [weight_before, weight_after]
        wind_direction_to_validate.loc[i] = circular_mean_wind_direction(direction_angles, weights)

    last_missing_index = missing_indices[-1]
    if wind_direction_to_validate.loc[last_missing_index] == -1:
        wind_direction_to_validate.loc[last_missing_index] = wind_direction_to_validate[last_missing_index - 2]

    data_frame['wind_direction'] = wind_direction_to_validate.astype(int)
    return data_frame


def get_validated_metar_data_frame():
    metar_df = metar_data_frame.get_metar_data_frame()
    return validate_wind_direction(metar_df)
