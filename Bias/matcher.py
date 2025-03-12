import DatabaseHandler


class Matcher:
    def __init__(self):
        self.database_handler = DatabaseHandler.DatabaseHandler()

    # match metars to the tafs based on the observation time
    # output: dictionary{ taf_id: list of matching metar ids }
    def match_metar_to_taf(self):
        tafs = self.database_handler.get_tafs_df()  # .sort_values('start_datetime')
        matched_data = {}
        self.database_handler.connect_to_database()

        for index, taf in tafs.iterrows():
            taf_id = taf.iloc[0]  # index of the id column in the tafs data frame
            start_time = str(taf.iloc[3])  # index of the start_time column in the tafs data frame
            end_time = str(taf.iloc[4])  # index of the end_time column in the tafs data frame
            metar_ids_df = self.database_handler.execute_query_return_list(
                """select id from metars where observation_time between""" + "'" + start_time + "'" + """and""" + "'" +
                end_time + "'")
            matched_data[taf_id] = metar_ids_df

        self.database_handler.close_connection()
        return matched_data

    # variabile de baza: predominant_horizontal_visibility, wind_speed,
    # wind_variability(true/false), wind_direction(can be null), weather_phenomena(can be null), clouds(can be null),
    # date in plus: sezon, ora din zi, air_temperature, dew_point, air_pressure
    # todo - o functie ce itereaza prin fiecare taf (fara probs) si care potriveste variabilele cu cele de la metar.
    #  functia verifica in prealabil daca taful contine si probs si le ia in considerare si pe acestea, verifica si daca
    #  metarul este in intervalul de timp al taf-ului

# print(match_metar_to_taf())
