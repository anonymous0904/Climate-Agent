import DatabaseHandler


# missing values can appear as None or NaN (not a number)
def get_metars_df():
    database_handler = DatabaseHandler.DatabaseHandler()
    database_handler.connect_to_database()
    metars_df = database_handler.execute_query("""SELECT * FROM metars""")
    database_handler.close_connection()
    return metars_df


# missing values can appear as None or NaN (not a number)
def get_tafs_df():
    database_handler = DatabaseHandler.DatabaseHandler()
    database_handler.connect_to_database()
    # fetch the tafs together with probable phenomena
    tafs_df = database_handler.execute_query("""SELECT * FROM tafs t, taf_probs tp where t.id = tp.taf_id""")
    database_handler.close_connection()
    return tafs_df


class LSTM:
    def __init__(self):
        self.metars_df = get_metars_df()
        self.tafs_df = get_tafs_df()


print(get_tafs_df())
