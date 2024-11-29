import datetime
import re

import psycopg2

wind_direction_speed = r'^[0-9]{5}KT$'
wind_direction_variation = r'^[0-9]{3}V[0-9]{3}$'
wind_gust = r'^[0-9]{5}G[0-9]{2}KT$'
variable_wind = r'^VRB[0-9]{2}KT$'
variable_wind_gust = r'^VRB[0-9]{2}G[0-9]{2}KT$'


def string_to_datetime(string):
    year = string[0:4]
    month = string[4:6]
    day = string[6:8]
    hour = string[8:10]
    minute = string[10:]
    datetime_string = year + '-' + month + '-' + day + ' ' + hour + ':' + minute + ":00"
    return datetime.datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")


def parse_wind(code_elements, final_vector):
    if re.match(wind_direction_speed, code_elements[0]):
        wind_direction = code_elements[0][0:3]
        wind_speed = code_elements[0][3:5]
        final_vector.append(int(wind_direction))
        final_vector.append(int(wind_speed))
        final_vector.append(False)  # no variability
        final_vector.append(None)  # gust speed (no gust)
        code_elements = code_elements[1:]
        if re.match(wind_direction_variation, code_elements[0]):  # direction variation after basic information
            low_direction = code_elements[0][0:3]
            up_direction = code_elements[0][4:]
            code_elements = code_elements[1:]
            final_vector.append(int(low_direction))
            final_vector.append(int(up_direction))
        else:  # no direction variation
            final_vector.append(None)
            final_vector.append(None)

    elif re.match(wind_gust, code_elements[0]):  # wind with gust
        wind_direction = code_elements[0][0:3]
        wind_speed = code_elements[0][3:5]
        gust_speed = code_elements[0][6:8]
        final_vector.append(int(wind_direction))
        final_vector.append(int(wind_speed))
        final_vector.append(False)  # no variability
        final_vector.append(int(gust_speed))
        code_elements = code_elements[1:]
        if re.match(wind_direction_variation, code_elements[0]):  # direction variation after gust information
            low_direction = code_elements[0][0:3]
            up_direction = code_elements[0][4:]
            code_elements = code_elements[1:]
            final_vector.append(int(low_direction))
            final_vector.append(int(up_direction))
        else:  # no direction variation
            final_vector.append(None)
            final_vector.append(None)

    elif re.match(variable_wind, code_elements[0]):
        final_vector.append(None)  # no concrete information about the direction (60-180 degrees)
        wind_speed = code_elements[0][3:5]
        final_vector.append(int(wind_speed))
        final_vector.append(True)  # wind variability
        final_vector.append(None)  # no gust
        final_vector.append(60)
        final_vector.append(180)  # when VRB the wind direction varies between 60 and 180
        code_elements = code_elements[1:]

    elif re.match(variable_wind_gust, code_elements[0]):  # variable wind with gust
        final_vector.append(None)  # no concrete information about the direction (60-180 degrees)
        wind_speed = code_elements[0][3:5]
        final_vector.append(int(wind_speed))
        final_vector.append(True)  # wind variability
        gust_speed = code_elements[0][6:8]
        final_vector.append(int(gust_speed))
        final_vector.append(60)
        final_vector.append(180)  # when VRB the wind direction varies between 60 and 180
        code_elements = code_elements[1:]
    return code_elements, final_vector


def present_phenomena(code_elements, final_vector):
    count = 0  # number of phenomena
    for i in range(3):
        if len(code_elements) == 0:
            break
        elif code_elements[0] == "NSW":
            code_elements = code_elements[1:]
            break
        elif re.match("^(\+|-|VC)", code_elements[0]) or re.match("^[A-Z]{2}$", code_elements[0]) or re.match(
                "[A-Z]{4}", code_elements[0]):
            final_vector.append(code_elements[0])
            code_elements = code_elements[1:]
            count += 1
        else:
            break

    while count < 3:  # for less than 3 phenomena
        final_vector.append(None)
        count += 1
    return code_elements, final_vector


def vertical_visibility(code_elements, final_vector):
    if len(code_elements) == 0:
        final_vector.append(None)
    elif re.match("^VV[0-9]{3}$", code_elements[0]):
        vv = code_elements[0][-3:]
        code_elements = code_elements[1:]
        final_vector.append(int(vv))
    else:
        final_vector.append(None)  # no vertical visibility
    return code_elements, final_vector


# returns the connection to the WeatherForecast database
def database_connection():
    try:
        db_connection = psycopg2.connect(
            host="localhost",
            database="WeatherForecast",
            user="postgres",
            password="castravete",
            port=5432
        )
        print("Connected to the database.")
        return db_connection
    except psycopg2.OperationalError as e:
        print(f"Error: {e}")
        return None


def time_period(date_time: datetime, string: str):
    start = string.split('/')[0]
    end = string.split('/')[1]
    start_year = date_time.year
    start_month = date_time.month
    start_day = start[0:2]
    start_hour = start[2:]
    end_year = date_time.year
    end_month = date_time.month
    end_day = end[0:2]
    end_hour = end[2:]

    #  start day is in the next month of the observation time
    if int(start_day) < date_time.day:
        # change year
        if date_time.month == 12 and date_time.day == 31:
            start_year = end_year = date_time.year + 1
            start_month = end_month = 1
        # change month if start time is on the first day
        elif (date_time.month in (1, 3, 5, 7, 8, 10) and date_time.day == 31) or (
                date_time.month in (4, 6, 9, 11) and date_time.day == 30) or (
                date_time.day == 28 and date_time.month == 2 and date_time.year % 4 != 0) or (
                date_time.day == 29 and date_time.month == 2 and date_time.year % 4 == 0):  # check for leap year
            start_month = end_month = date_time.month + 1

    # different days for the start and end time
    elif int(start_day) > int(end_day):
        # change year
        if date_time.month == 12 and date_time.day == 31:
            end_year = date_time.year + 1
            end_month = 1
        # change month if end time is on the first day of the next month
        elif (date_time.month in (1, 3, 5, 7, 8, 10) and date_time.day == 31) or (
                date_time.month in (4, 6, 9, 11) and date_time.day == 30) or (
                date_time.day == 28 and date_time.month == 2 and date_time.year % 4 != 0) or (
                date_time.day == 29 and date_time.month == 2 and date_time.year % 4 == 0):  # check for leap year
            end_month = date_time.month + 1

    start_datetime_string = str(start_year) + '-' + str(start_month) + '-' + start_day + ' ' + start_hour + ":00:00"
    start_datetime = datetime.datetime.strptime(start_datetime_string, "%Y-%m-%d %H:%M:%S")
    end_datetime_string = str(end_year) + '-' + str(end_month) + '-' + end_day + ' ' + end_hour + ":00:00"
    end_datetime = datetime.datetime.strptime(end_datetime_string, "%Y-%m-%d %H:%M:%S")
    return start_datetime, end_datetime


def disconnect_from_database(connection):
    connection.close()
    print("Disconnected from the database.")
