import DatabaseHandler


# variabile de baza: predominant_horizontal_visibility, wind_direction(can be null), wind_speed,
# wind_variability(true/false), cavok, present_phenomena 1, 2 and 3 (can be null), cloud_nebulosity(can be null),
# cloud_altitude(can be null), presence of cumulonimbus(true/false), vertical visibility(most of them are null)
# date in plus: sezon, ora din zi, air_temperature, dew_point, air_pressure
class Bias:

    def __init__(self):
        self.database_handler = DatabaseHandler.DatabaseHandler()
        self.tafs_metars_df = self.database_handler.get_taf_probs_metars_df()
