import re

import Parser.parser
from Parser import parser
from Parser.parser import string_to_datetime, parse_wind, database_connection, disconnect_from_database


# metar is a vector with elements from the code,
# it inserts each variable to the corresponding fields of the table
def add_metar_into_table(metar):
    insert_query = """INSERT INTO metars (observation_time, message_type, message_callsign, wind_direction, 
    wind_speed, wind_variability, gust_speed, lower_direction_variation, upper_direction_variation, cavok, 
    predominant_horizontal_visibility,directional_horizontal_visibility, directional_variation_visibility,
    runway_number,runway_visibility,visibility_indicator,tendency,minimal_runway_visibility,maximal_runway_visibility,
    minimal_indicator,maximal_indicator,present_phenomena_1,present_phenomena_2,present_phenomena_3,cloud_nebulosity_1,
    cloud_altitude_1,cloud_type_1,cloud_nebulosity_2,cloud_altitude_2,cloud_type_2,cloud_nebulosity_3,cloud_altitude_3,
    cloud_type_3,no_cloud,vertical_visibility,air_temperature,dew_point,air_pressure) 
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    query_values = tuple(metar)
    cursor.execute(insert_query, query_values)
    connection.commit()


# returns a vector of the variables from the metar(one single metar at a time)
# if a variable is not present, it is saved as None element in the vector
def parse_metar(metar):
    metar_elements = metar.rstrip('=').split(' ')
    final_metar_vector = []

    # observation time
    final_metar_vector.append(string_to_datetime(metar_elements[0]))
    metar_elements = metar_elements[1:]

    # message type - METAR/ METAR COR/ SPECI/ SPECI COR
    if metar_elements[1] == "COR":  # check if the message is corrected
        final_metar_vector.append(metar_elements[0] + ' ' + metar_elements[1])
        metar_elements = metar_elements[2:]
    else:
        final_metar_vector.append(metar_elements[0])
        metar_elements = metar_elements[1:]

    # get rid of ICAO code and redundant observation time
    metar_elements = metar_elements[2:]

    # check for message callsign(AUTO/NIL)
    try:
        if len(metar_elements) == 0:  # NIL
            return final_metar_vector
        elif metar_elements[0] == "NIL":
            final_metar_vector.append(metar_elements[0])
            return final_metar_vector
        elif metar_elements[0] == "AUTO":
            final_metar_vector.append(metar_elements[0])
            metar_elements = metar_elements[1:]
        else:
            final_metar_vector.append(None)
    except IndexError as e:
        print(e.args, ':', metar)

    # wind
    metar_elements, final_metar_vector = parse_wind(metar_elements, final_metar_vector)

    # CAVOK
    if metar_elements[0] == "CAVOK":
        final_metar_vector.append(True)  # CAVOK true
        final_metar_vector.append(9999)  # horizontal visibility
        final_metar_vector = final_metar_vector + 24 * [
            None]  # no clouds, no presente phenomena, no visibility variation
        metar_elements = metar_elements[1:]

    else:
        # predominant horizontal visibility
        try:
            final_metar_vector.append(False)  # no CAVOK
            visibility = metar_elements[0][0:4]
            final_metar_vector.append(int(visibility))
            if re.match("^[0-9]{4}NDV$", metar_elements[0]):  # no information about directional variation
                final_metar_vector.append(None)  # no directional visibility
                final_metar_vector.append("NDV")
                metar_elements = metar_elements[1:]
        except Exception as e:
            print(e.args, '+', metar)
            return final_metar_vector  # invalid code

        else:  # check for directional visibility
            metar_elements = metar_elements[1:]
            if re.match("^[0-9]{4}(N|S|E|V|W|SE|SV|NV|NE|NW|SW)$", metar_elements[0]):
                final_metar_vector.append(int(metar_elements[0][0:4]))
                final_metar_vector.append(metar_elements[0][4:])
                metar_elements = metar_elements[1:]
            else:  # no values for directional horizontal visibility and variation
                final_metar_vector.append(None)
                final_metar_vector.append(None)

        # runway visibility
        if re.match("^R[0-9]{2}(L|LL|R|RR|C)?/[MP]?[0-9]{4}V?[NUD]?$", metar_elements[0]):
            runway_number = metar_elements[0][1:3]
            final_metar_vector.append(int(runway_number))
            visibility_group = metar_elements[0].split('/')[1]  # takes the string after '/'
            visibility_indicator = None
            # check for values that don't fit the measurements of the system
            if visibility_group[0] == 'P' or visibility_group[0] == 'M':
                visibility_indicator = visibility_group[0]
                visibility_group = visibility_group[1:]

            runway_visibility = visibility_group[0:4]
            visibility_tendency = None
            if visibility_group[-1] == 'N' or visibility_group[-1] == 'U' or visibility_group[-1] == 'D':
                visibility_tendency = visibility_group[-1]

            final_metar_vector.append(int(runway_visibility))
            final_metar_vector.append(visibility_indicator)
            final_metar_vector.append(visibility_tendency)
            final_metar_vector = final_metar_vector + 4 * [None]
            metar_elements = metar_elements[1:]

        elif re.match("^R[0-9]{2}(L|LL|R|RR|C)?/[MP]?[0-9]{4}V[MP]?[0-9]{4}[NUD]?$", metar_elements[0]):
            runway_number = metar_elements[0][1:3]
            final_metar_vector.append(int(runway_number))
            visibility_group = metar_elements[0].split('/')[1]  # takes the string after '/'
            runway_visibility = None
            visibility_indicator = None
            visibility_tendency = None
            minimal_indicator = None
            maximal_indicator = None
            if visibility_group[-1] == 'N' or visibility_group[-1] == 'U' or visibility_group[-1] == 'D':
                visibility_tendency = visibility_group[-1]

            if visibility_group[0] == 'M' or visibility_group[0] == 'P':
                minimal_indicator = visibility_group[0]
                visibility_group = visibility_group[1:]

            minimal_runway_visibility = visibility_group[0:4]
            visibility_group = visibility_group[5:]  # without 'V'

            if visibility_group[0] == 'M' or visibility_group[0] == 'P':
                maximal_indicator = visibility_group[0]
                visibility_group = visibility_group[1:]

            maximal_runway_visibility = visibility_group[0:4]
            final_metar_vector = final_metar_vector + [runway_visibility, visibility_indicator, visibility_tendency,
                                                       int(minimal_runway_visibility),
                                                       int(maximal_runway_visibility), minimal_indicator,
                                                       maximal_indicator]
            metar_elements = metar_elements[1:]

        else:  # no information about runway visibility
            final_metar_vector = final_metar_vector + 8 * [None]

        # present phenomena - up to 3
        metar_elements, final_metar_vector = parser.present_phenomena(metar_elements, final_metar_vector)

        # Clouds
        count = 0  # number of layers
        clouds_detected = False
        for i in range(3):
            if re.match("^(FEW|SCT|BKN|OVC)", metar_elements[0]):
                count += 1
                nebulosity = metar_elements[0][0:3]
                altitude = metar_elements[0][3:6]
                cloud_type = None

                if metar_elements[0][-3:] == "///" or metar_elements[0][-3:] == "TCU":
                    cloud_type = metar_elements[0][-3:]
                if metar_elements[0][-2:] == "CB":
                    cloud_type = metar_elements[0][-2:]

                metar_elements = metar_elements[1:]
                final_metar_vector.append(nebulosity)
                final_metar_vector.append(int(altitude))
                final_metar_vector.append(cloud_type)

            elif re.match("^//////", metar_elements[0]):
                count += 1
                nebulosity = altitude = cloud_type = None
                if metar_elements[0][-3:] == "TCU":
                    cloud_type = metar_elements[0][-3:]
                elif metar_elements[0][-2:] == "CB":
                    cloud_type = metar_elements[0][-2:]

                metar_elements = metar_elements[1:]
                final_metar_vector.append(nebulosity)
                final_metar_vector.append(altitude)
                final_metar_vector.append(cloud_type)
            else:
                break
        if count > 0:
            clouds_detected = True
        while count < 3:
            final_metar_vector = final_metar_vector + 3 * [None]  # for nebulosity, altitude and cloud_type
            count += 1

        if clouds_detected is True:
            final_metar_vector.append(None)  # no NCD or NSC
        # NSC or NCD
        elif metar_elements[0] == "NCD" or metar_elements[0] == "NSC":
            final_metar_vector.append(metar_elements[0])
            metar_elements = metar_elements[1:]
        else:
            final_metar_vector.append(None)

        # vertical visibility
        metar_elements, final_metar_vector = parser.vertical_visibility(metar_elements, final_metar_vector)

    # End of no CAVOK

    # Air temperature/ dew point
    if re.match("^M?[0-9]{2}/M?[0-9]{2}$", metar_elements[0]):
        if metar_elements[0][0] == 'M':
            air_temperature = -int(metar_elements[0][1:3])
        else:
            air_temperature = int(metar_elements[0][0:2])
        dew_point_group = metar_elements[0].split('/')[1]
        if dew_point_group[0] == 'M':
            dew_point = -int(dew_point_group[1:])
        else:
            dew_point = int(dew_point_group[0:])

        final_metar_vector.append(air_temperature)
        final_metar_vector.append(dew_point)
        metar_elements = metar_elements[1:]

    # Air pressure
    if re.match("^Q[0-9]{4}", metar_elements[0]):
        final_metar_vector.append(int(metar_elements[0][1:5]))
        metar_elements = metar_elements[1:]

    return final_metar_vector


connection = database_connection()
cursor = connection.cursor()


# reads every line and calls other function to parse each of them
def metar_parser(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    for line in reversed(lines):
        parsed_metar = parse_metar(line)
        if len(parsed_metar) == 38:
            add_metar_into_table(parsed_metar)
    file.close()
    disconnect_from_database(connection)
