import DatabaseHandler
from errors import *


# variabile de baza: predominant_horizontal_visibility, wind_direction(can be null), wind_speed,
# wind_variability(true/false), cavok, present_phenomena 1, 2 and 3 (can be null), cloud_nebulosity(can be null),
# cloud_altitude(can be null), presence of cumulonimbus(true/false), vertical visibility(most of them are null)
# date in plus: sezon, ora din zi, air_temperature, dew_point, air_pressure
class Bias:

    def __init__(self):
        self.database_handler = DatabaseHandler.DatabaseHandler()
        self.tafs_metars_df = self.database_handler.get_taf_probs_metars_df()

    def apply_errors_to_dataframe(self):
        self.tafs_metars_df['month'] = self.tafs_metars_df.apply(get_month_of_the_row, axis=1)
        self.tafs_metars_df['season'] = self.tafs_metars_df.apply(get_season_of_the_row, axis=1)
        self.tafs_metars_df['taf_wind_direction_error'] = self.tafs_metars_df.apply(
            calculate_taf_wind_direction_error_for_row, axis=1)
        self.tafs_metars_df['taf_prob_wind_direction_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_wind_direction_error_for_row, axis=1)
        self.tafs_metars_df['taf_wind_speed_error'] = self.tafs_metars_df.apply(calculate_taf_wind_speed_error_for_row,
                                                                                axis=1)
        self.tafs_metars_df['taf_prob_wind_speed_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_wind_speed_error_for_row, axis=1)
        self.tafs_metars_df['taf_wind_variability_error'] = self.tafs_metars_df.apply(
            calculate_taf_wind_variability_error_for_row, axis=1)
        self.tafs_metars_df['taf_prob_wind_variability_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_wind_variability_error_for_row, axis=1)
        self.tafs_metars_df['taf_gust_speed_error'] = self.tafs_metars_df.apply(calculate_taf_gust_speed_error_for_row,
                                                                                axis=1)
        self.tafs_metars_df['taf_prob_gust_speed_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_gust_speed_error_for_row, axis=1)
        self.tafs_metars_df['taf_horizontal_visibility_error'] = self.tafs_metars_df.apply(
            calculate_taf_horizontal_visibility_error_for_row, axis=1)
        self.tafs_metars_df['taf_prob_horizontal_visibility_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_horizontal_visibility_error_for_row, axis=1)
        self.tafs_metars_df['taf_rainfall_error'] = self.tafs_metars_df.apply(calculate_taf_rainfall_error_for_row,
                                                                              axis=1)
        self.tafs_metars_df['taf_prob_rainfall_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_rainfall_error_for_row, axis=1)
        self.tafs_metars_df['taf_fog_error'] = self.tafs_metars_df.apply(calculate_taf_fog_error_for_row, axis=1)
        self.tafs_metars_df['taf_prob_fog_error'] = self.tafs_metars_df.apply(calculate_taf_prob_fog_error_for_row,
                                                                              axis=1)
        self.tafs_metars_df['taf_cloud_nebulosity_error'] = self.tafs_metars_df.apply(
            calculate_taf_cloud_nebulosity_error_for_row, axis=1)
        self.tafs_metars_df['taf_prob_cloud_nebulosity_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_cloud_nebulosity_error_for_row, axis=1)
        self.tafs_metars_df['taf_cloud_altitude_error'] = self.tafs_metars_df.apply(
            calculate_taf_lowest_cloud_altitude_error_for_row, axis=1)
        self.tafs_metars_df['taf_prob_cloud_altitude_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_lowest_cloud_altitude_error_for_row, axis=1)
        self.tafs_metars_df['taf_number_cloud_layers_error'] = self.tafs_metars_df.apply(
            calculate_taf_cloud_layers_error_for_row, axis=1)
        self.tafs_metars_df['taf_prob_number_cloud_layers_error'] = self.tafs_metars_df.apply(
            calculate_taf_prob_cloud_layers_error_for_row, axis=1)

    def print_data_frame(self):
        print(self.tafs_metars_df)


bias = Bias()
bias.apply_errors_to_dataframe()
bias.print_data_frame()
