import psycopg2
import pandas as pd


# class to handle the connection and data retrieval from the WeatherForecast database
class DatabaseHandler:
    def __init__(self):
        self.connection = None

    def connect_to_database(self):
        try:
            self.connection = psycopg2.connect(
                host="localhost",
                database="WeatherForecast",
                user="postgres",
                password="castravete",
                port=5432
            )
            print("Connected to the database.")
        except psycopg2.OperationalError as e:
            print(f"Error: {e}")

    def execute_query_return_df(self, query):
        if not self.connection:
            print('Not connected to WeatherForecast database')
            return
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            cursor.close()
            return pd.DataFrame(data, columns=column_names)
        except psycopg2.Error:
            print(f'Error executing query')

    def execute_query_return_list(self, query):
        if not self.connection:
            print('Not connected to WeatherForecast database')
            return
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            return data
        except psycopg2.Error:
            print(f'Error executing query')

    def close_connection(self):
        self.connection.close()
        print("Disconnected from the database.")

    # fetch the metars from the database
    def get_metars_df(self):
        self.connect_to_database()
        # missing values can appear as None or NaN (not a number)
        metars_df = self.execute_query_return_df("""SELECT * FROM metars""")
        self.close_connection()
        return metars_df

    # fetch the tafs together with probable phenomena
    def get_tafs_with_probs_df(self):
        self.connect_to_database()
        # missing values can appear as None or NaN (not a number)
        tafs_df = self.execute_query_return_df("""SELECT * FROM tafs t, taf_probs tp where t.id = tp.taf_id""")
        self.close_connection()
        return tafs_df

    # fetch the tafs from the database
    def get_tafs_df(self):
        self.connect_to_database()
        tafs_df = self.execute_query_return_df("""SELECT * FROM tafs""")
        self.close_connection()
        return tafs_df

    # fetch the tafs together with the corresponding metars based on the observation time
    def get_tafs_metars_df(self):
        self.connect_to_database()
        tafs_metars_df = self.execute_query_return_df("""SELECT * FROM tafs t, metars m where 
                                       m.observation_time between t.start_datetime and t.end_datetime""")
        self.close_connection()
        return tafs_metars_df

    # fetch the taf probs together with the corresponding metars based on the observation time
    def get_taf_probs_metars_df(self):
        self.connect_to_database()
        taf_probs_metars_df = self.execute_query_return_df(
            """SELECT * FROM taf_probs tp, metars m where m.observation_time between tp.start_time and tp.end_time""")
        self.close_connection()
        return taf_probs_metars_df

    # fetch all the tafs joined with taf_probs (including those without probable phenomena) joined with metars
    # based on the observation time
    def get_taf_with_probs_and_metars_df(self):
        self.connect_to_database()
        taf_with_probs_metars_df = self.execute_query_return_df("""select * from tafs t left join taf_probs tp on 
        t.id=tp.taf_id inner join metars m on m.observation_time between t.start_datetime and t.end_datetime 
        order by t.id""")
        self.close_connection()
        return taf_with_probs_metars_df


print(DatabaseHandler().get_taf_with_probs_and_metars_df().columns)
