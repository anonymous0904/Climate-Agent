import re

from Parser import parser


# connection = parser.database_connection()
# cursor = connection.cursor()


def parse_taf(taf):
    final_taf_vector = []
    taf_elements = taf.split(' ')

    # forecast time
    try:
        datetime = parser.string_to_datetime(taf_elements[0])
        final_taf_vector.append(datetime)
        taf_elements = taf_elements[1:]
    except Exception as e:
        print(e.args, ':' + taf)
        return final_taf_vector

    # message type: TAF/ TAF COR/ TAF AMD
    try:
        if taf_elements[1] == "COR" or taf_elements[1] == "AMD":  # check for COR or AMD
            final_taf_vector.append(taf_elements[0] + ' ' + taf_elements[1])
            taf_elements = taf_elements[2:]
        else:
            final_taf_vector.append(taf_elements[0])
            taf_elements = taf_elements[1:]
    except Exception as e:
        print(e.args, ':' + taf)
        return final_taf_vector

    # get rid of ICAO code and redundant observation time
    try:
        taf_elements = taf_elements[2:]
    except Exception as e:
        print(e.args, ':' + taf)
        return final_taf_vector

    # check for NIL
    if len(taf_elements) == 0 or taf_elements[0] == "NIL":
        return final_taf_vector

    # time period of the forecasted weather
    try:
        start_time, end_time = parser.time_period(datetime, taf_elements[0])
        taf_elements = taf_elements[1:]
        final_taf_vector.append(start_time)
        final_taf_vector.append(end_time)
    except Exception as e:
        print(e.args, ':' + taf)
        return final_taf_vector

    # wind
    # taf_elements, final_taf_vector = parser.parse_wind(taf_elements, final_taf_vector)
    try:
        if re.match(parser.wind_direction_speed, taf_elements[0]):
            wind_direction = taf_elements[0][0:3]
            wind_speed = taf_elements[0][3:5]
            final_taf_vector.append(int(wind_direction))
            final_taf_vector.append(int(wind_speed))
            final_taf_vector.append(False)  # no variability
            final_taf_vector.append(None)  # gust speed (no gust)
            taf_elements = taf_elements[1:]
        elif re.match(parser.wind_gust, taf_elements[0]):  # wind with gust
            wind_direction = taf_elements[0][0:3]
            wind_speed = taf_elements[0][3:5]
            gust_speed = taf_elements[0][6:8]
            final_taf_vector.append(int(wind_direction))
            final_taf_vector.append(int(wind_speed))
            final_taf_vector.append(False)  # no variability
            final_taf_vector.append(int(gust_speed))
            taf_elements = taf_elements[1:]
        elif re.match(parser.variable_wind, taf_elements[0]):
            final_taf_vector.append(None)  # no concrete information about the direction (60-180 degrees)
            wind_speed = taf_elements[0][3:5]
            final_taf_vector.append(int(wind_speed))
            final_taf_vector.append(True)  # wind variability
            final_taf_vector.append(None)  # no gust
            taf_elements = taf_elements[1:]
        elif re.match(parser.variable_wind_gust, taf_elements[0]):  # variable wind with gust
            final_taf_vector.append(None)  # no concrete information about the direction (60-180 degrees)
            wind_speed = taf_elements[0][3:5]
            final_taf_vector.append(int(wind_speed))
            final_taf_vector.append(True)  # wind variability
            gust_speed = taf_elements[0][6:8]
            final_taf_vector.append(int(gust_speed))
            taf_elements = taf_elements[1:]
    except Exception as e:
        print(e.args, ':' + taf)
        return final_taf_vector

    # CAVOK
    try:
        if taf_elements[0] == "CAVOK":
            final_taf_vector.append(True)  # CAVOK true
            final_taf_vector.append(9999)  # horizontal visibility
            final_taf_vector = final_taf_vector + 5 * [
                None] + [False] + 2 * [None, None, False] + [None]  # no clouds, no present phenomena
            taf_elements = taf_elements[1:]

        else:
            # predominant horizontal visibility
            try:
                final_taf_vector.append(False)  # no CAVOK
                visibility = taf_elements[0]
                final_taf_vector.append(int(visibility))
                taf_elements = taf_elements[1:]
            except Exception as e:
                print(e.args, '+', taf)
                return final_taf_vector  # invalid code

            # Present phenomena - up to 3
            taf_elements, final_taf_vector = parser.present_phenomena(taf_elements, final_taf_vector)

            # Clouds
            count = 0  # number of layers
            clouds_detected = False
            for i in range(3):
                if len(taf_elements) == 0:
                    break
                elif taf_elements[0] == "NSC":
                    taf_elements = taf_elements[1:]
                    break
                elif re.match("^(FEW|SCT|BKN|OVC)", taf_elements[0]):
                    count += 1
                    nebulosity = taf_elements[0][0:3]
                    altitude = taf_elements[0][3:6]
                    cb = False

                    if taf_elements[0][-2:] == "CB":
                        cb = True

                    taf_elements = taf_elements[1:]
                    final_taf_vector.append(nebulosity)
                    final_taf_vector.append(int(altitude))
                    final_taf_vector.append(cb)

                else:
                    break

            while count < 3:
                final_taf_vector = final_taf_vector + [None, None,
                                                       False]  # for nebulosity, altitude and presence of cumulonimbus
                count += 1

            # Vertical visibility
            taf_elements, final_taf_vector = parser.vertical_visibility(taf_elements, final_taf_vector)

        return final_taf_vector
    except Exception as e:
        print(e.args, ':' + taf)
        return final_taf_vector


#  weather forecast with a given probability
def parse_taf_prob(datetime, taf_prob):
    final_taf_prob_vector = []
    try:
        taf_prob_elements = taf_prob.split(' ')
        if re.match("^PROB..$", taf_prob_elements[0]):
            probability = taf_prob_elements[0][4:]
            final_taf_prob_vector.append(int(probability))
            if taf_prob_elements[1] == "TEMPO" or taf_prob_elements[1] == "BECMG":
                final_taf_prob_vector.append(taf_prob_elements[1])
                taf_prob_elements = taf_prob_elements[2:]
            else:
                final_taf_prob_vector.append("PROB")
                taf_prob_elements = taf_prob_elements[1:]
        else:
            final_taf_prob_vector.append(None)  # no probability given
            final_taf_prob_vector.append(taf_prob_elements[0])  # BECMG or TEMPO
            taf_prob_elements = taf_prob_elements[1:]

        # time period of the forecasted weather
        start_time, end_time = parser.time_period(datetime, taf_prob_elements[0])
        taf_prob_elements = taf_prob_elements[1:]
        final_taf_prob_vector.append(start_time)
        final_taf_prob_vector.append(end_time)

        # check for NIL
        if len(taf_prob_elements) == 0:
            return final_taf_prob_vector

        # wind
        # taf_prob_elements, final_taf_prob_vector = parser.parse_wind(taf_prob_elements, final_taf_prob_vector)
        if re.match(parser.wind_direction_speed, taf_prob_elements[0]):
            wind_direction = taf_prob_elements[0][0:3]
            wind_speed = taf_prob_elements[0][3:5]
            final_taf_prob_vector.append(int(wind_direction))
            final_taf_prob_vector.append(int(wind_speed))
            final_taf_prob_vector.append(False)  # no variability
            final_taf_prob_vector.append(None)  # gust speed (no gust)
            taf_prob_elements = taf_prob_elements[1:]
        elif re.match(parser.wind_gust, taf_prob_elements[0]):  # wind with gust
            wind_direction = taf_prob_elements[0][0:3]
            wind_speed = taf_prob_elements[0][3:5]
            gust_speed = taf_prob_elements[0][6:8]
            final_taf_prob_vector.append(int(wind_direction))
            final_taf_prob_vector.append(int(wind_speed))
            final_taf_prob_vector.append(False)  # no variability
            final_taf_prob_vector.append(int(gust_speed))
            taf_prob_elements = taf_prob_elements[1:]
        elif re.match(parser.variable_wind, taf_prob_elements[0]):
            final_taf_prob_vector.append(None)  # no concrete information about the direction (60-180 degrees)
            wind_speed = taf_prob_elements[0][3:5]
            final_taf_prob_vector.append(int(wind_speed))
            final_taf_prob_vector.append(True)  # wind variability
            final_taf_prob_vector.append(None)  # no gust
            taf_prob_elements = taf_prob_elements[1:]
        elif re.match(parser.variable_wind_gust, taf_prob_elements[0]):  # variable wind with gust
            final_taf_prob_vector.append(None)  # no concrete information about the direction (60-180 degrees)
            wind_speed = taf_prob_elements[0][3:5]
            final_taf_prob_vector.append(int(wind_speed))
            final_taf_prob_vector.append(True)  # wind variability
            gust_speed = taf_prob_elements[0][6:8]
            final_taf_prob_vector.append(int(gust_speed))
            taf_prob_elements = taf_prob_elements[1:]
        else:  # no wind
            final_taf_prob_vector = final_taf_prob_vector + [None, None, False, None]

        if len(taf_prob_elements) == 0:
            final_taf_prob_vector = final_taf_prob_vector + [False] + 4 * [None] + 3 * [None, None, False] + [None]
            return final_taf_prob_vector
        # CAVOK
        elif taf_prob_elements[0] == "CAVOK":
            final_taf_prob_vector.append(True)  # CAVOK true
            final_taf_prob_vector.append(9999)  # horizontal visibility
            final_taf_prob_vector = final_taf_prob_vector + 5 * [
                None] + [False] + 2 * [None, None, False] + [None]  # no clouds, no present phenomena
            taf_prob_elements = taf_prob_elements[1:]

        else:
            # predominant horizontal visibility
            try:
                final_taf_prob_vector.append(False)  # no CAVOK
                if len(taf_prob_elements) == 0:  # no forecast regarding further phenomena
                    final_taf_prob_vector = final_taf_prob_vector + 4 * [None] + 3 * [None, None, False] + [None]
                    return final_taf_prob_vector
                elif re.match("^[0-9]{4}$", taf_prob_elements[0]):
                    visibility = taf_prob_elements[0]
                    final_taf_prob_vector.append(int(visibility))
                    taf_prob_elements = taf_prob_elements[1:]
                else:  # no forecast regarding visibility
                    final_taf_prob_vector.append(None)
            except Exception as e:
                print(e.args, '+', taf_prob)
                return final_taf_prob_vector  # invalid code

            # Present phenomena - up to 3
            taf_prob_elements, final_taf_prob_vector = parser.present_phenomena(taf_prob_elements,
                                                                                final_taf_prob_vector)

            # Clouds
            count = 0  # number of layers
            clouds_detected = False
            for i in range(3):
                if len(taf_prob_elements) == 0:
                    break
                elif taf_prob_elements[0] == "NSC":
                    taf_prob_elements = taf_prob_elements[1:]
                    break
                elif re.match("^(FEW|SCT|BKN|OVC)", taf_prob_elements[0]):
                    count += 1
                    nebulosity = taf_prob_elements[0][0:3]
                    altitude = taf_prob_elements[0][3:6]
                    cb = False

                    if taf_prob_elements[0][-2:] == "CB":
                        cb = True

                    taf_prob_elements = taf_prob_elements[1:]
                    final_taf_prob_vector.append(nebulosity)
                    final_taf_prob_vector.append(int(altitude))
                    final_taf_prob_vector.append(cb)

                else:
                    break

            while count < 3:
                final_taf_prob_vector = final_taf_prob_vector + [None, None,
                                                                 False]  # for nebulosity, altitude and presence of
                # cumulonimbus
                count += 1

            # Vertical visibility
            taf_prob_elements, final_taf_prob_vector = parser.vertical_visibility(taf_prob_elements,
                                                                                  final_taf_prob_vector)

        return final_taf_prob_vector
        # final_taf_prob_vector = final_taf_prob_vector + parse_taf(' '.join(taf_prob_elements))
    except Exception as e:
        return final_taf_prob_vector


# returns a list of strings with the main taf and the probs separately
def parse_code(taf):
    taf_elements = list(map(lambda x: x.rstrip('\n'), [elem for elem in taf.rstrip('=').split(' ') if
                                                       elem != '']))  # get rid of potential newline characters
    taf_vector = []
    i = len(taf_elements) - 1
    while i >= 0:
        if taf_elements[i] == "TEMPO" or taf_elements[i] == "BECMG":
            if re.match("^PROB..$", taf_elements[i - 1]):  # if TEMPO or BECMG have a specific probability
                i -= 1
            taf_vector.insert(0, ' '.join(taf_elements[i:]))  # insert at the beginning of the list
            taf_elements = taf_elements[:i]
        elif taf_elements[i] == "PROB30" or taf_elements[i] == "PROB40":
            taf_vector.insert(0, ' '.join(taf_elements[i:]))
            taf_elements = taf_elements[:i]
        elif i == 0:
            taf_vector.insert(0, ' '.join(taf_elements))
        i -= 1
    return taf_vector


# def add_taf_to_table(taf):
#     insert_query = """INSERT INTO tafs (datetime,message_type, start_datetime,end_datetime,wind_direction,wind_speed,
#     wind_variability,gust_speed,cavok,horizontal_visibility,
#     present_phenomena_1,present_phenomena_2,present_phenomena_3,cloud_nebulosity_1,cloud_altitude_1,cb_1,
#     cloud_nebulosity_2,cloud_altitude_2,cb_2,cloud_nebulosity_3,cloud_altitude_3,cb_3,
#     vertical_visibility)
#     values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
#     query_values = tuple(taf)
#     cursor.execute(insert_query, query_values)
#     connection.commit()


# def add_taf_prob_to_table(datetime, taf_prob):
#     select_query = f"SELECT id FROM tafs WHERE datetime='{datetime}'"
#     cursor.execute(select_query)
#     taf_id = cursor.fetchall()[0]
#     query_values = tuple([taf_id] + taf_prob)
#     insert_query = """INSERT INTO taf_probs( taf_id, probability, probability_type, start_time, end_time,
#     wind_direction, wind_speed, wind_variability, gust_speed,
#     cavok, horizontal_visibility, present_phenomena_1, present_phenomena_2, present_phenomena_3, cloud_nebulosity_1,
#     cloud_altitude_1, cb_1, cloud_nebulosity_2, cloud_altitude_2, cb_2, cloud_nebulosity_3,
#     cloud_altitude_3, cb_3, vertical_visibility) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
#     %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) """
#     cursor.execute(insert_query, query_values)
#     connection.commit()


def taf_parser(filename):
    file = open(filename, 'r')
    content = file.read()
    taf_codes = content.replace('\n', ' ').split('=')[:-1]  # remove last element which is an empty string
    for taf_code in taf_codes:
        taf_elements = parse_code(taf_code)
        parsed_taf = parse_taf(taf_elements[0])
        if len(parsed_taf) != 23:
            print(taf_elements[0])
            # add_taf_to_table(parsed_taf)
        datetime = parser.string_to_datetime(taf_elements[0][0:12])
        taf_elements = taf_elements[1:]
        for taf_prob in taf_elements:
            parsed_taf_prob = parse_taf_prob(datetime, taf_prob)
            if len(parsed_taf_prob) != 23:
                print(taf_prob)
                # add_taf_prob_to_table(datetime,parsed_taf_prob)

    # for i in range(100):
    #     print(taf_codes[i])

    file.close()
    # cursor.close()
    # parser.disconnect_from_database(connection)


taf_parser("tafs.txt")
# s = """202309241100 TAF LRCL 241100Z 2412/2421 VRB04KT 9999 SCT050
#                       TEMPO 2412/2415 VRB15G25KT 5000 TSRA BKN010
#                        BKN030CB
#                       BECMG 2413/2415 30010KT="""
# print(s.replace('\n', ' '))
