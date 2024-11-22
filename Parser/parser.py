import datetime
import re

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
