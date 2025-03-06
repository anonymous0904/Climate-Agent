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

    def execute_query(self, query):
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

    def close_connection(self):
        self.connection.close()
        print("Disconnected from the database.")
