import datetime
import unittest
from Parser import taf_parser


class TafParserTest(unittest.TestCase):

    def testParseCodeNoProbs(self):
        actual_result = taf_parser.parse_code("202205130200 TAF LRCL 130200Z 1303/1312 VRB04KT CAVOK=")
        expected_result = ["202205130200 TAF LRCL 130200Z 1303/1312 VRB04KT CAVOK"]
        self.assertEqual(actual_result, expected_result)

    def testParseCodeOneProb(self):
        actual_result = taf_parser.parse_code(
            "202205131100 TAF LRCL 131100Z 1312/1321 VRB04KT CAVOK                       PROB30 1312/1317 VRB18G28KT "
            "5000 TSRA=")
        expected_result = ["202205131100 TAF LRCL 131100Z 1312/1321 VRB04KT CAVOK",
                           "PROB30 1312/1317 VRB18G28KT 5000 TSRA"]
        self.assertEqual(actual_result, expected_result)

    def testParseCodeOneBECMG(self):
        actual_result = taf_parser.parse_code(
            "202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK                       BECMG 1407/1409 33012KT=")
        expected_result = ["202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK",
                           "BECMG 1407/1409 33012KT"]
        self.assertEqual(actual_result, expected_result)

    def testParseCodeOneTEMPO(self):
        actual_result = taf_parser.parse_code(
            "202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK                       TEMPO 1407/1409 33012KT=")
        expected_result = ["202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK",
                           "TEMPO 1407/1409 33012KT"]
        self.assertEqual(actual_result, expected_result)

    def testParseCodeOneProbTEMPO(self):
        actual_result = taf_parser.parse_code(
            "202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK                       PROB40 TEMPO 1407/1409 "
            "33012KT=")
        expected_result = ["202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK",
                           "PROB40 TEMPO 1407/1409 33012KT"]
        self.assertEqual(actual_result, expected_result)

    def testParseCodeOneProbBECMG(self):
        actual_result = taf_parser.parse_code(
            "202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK                       PROB40 BECMG 1407/1409 "
            "33012KT=")
        expected_result = ["202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK",
                           "PROB40 BECMG 1407/1409 33012KT"]
        self.assertEqual(actual_result, expected_result)

    def testParseCodeTempoBECMG(self):
        actual_result = taf_parser.parse_code(
            "202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK                       TEMPO 1407/1409 33012KT      "
            "                 BECMG 1407/1409 "
            "33012KT=")
        expected_result = ["202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK", "TEMPO 1407/1409 33012KT",
                           "BECMG 1407/1409 33012KT"]
        self.assertEqual(actual_result, expected_result)

    def testParseCodeProbTempoBECMG(self):
        actual_result = taf_parser.parse_code(
            "202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK                       PROB30 1312/1317 VRB18G28KT  "
            "                     TEMPO 1407/1409 33012KT "
            "                 BECMG 1407/1409 "
            "33012KT=")
        expected_result = ["202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK", "PROB30 1312/1317 VRB18G28KT",
                           "TEMPO 1407/1409 33012KT",
                           "BECMG 1407/1409 33012KT"]
        self.assertEqual(actual_result, expected_result)

    def testParseCodeProbProbsTempoBECMG(self):
        actual_result = taf_parser.parse_code(
            "202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK                       PROB30 1312/1317 VRB18G28KT  "
            "                     PROB40 TEMPO 1407/1409 33012KT "
            "                 PROB30 BECMG 1407/1409 "
            "33012KT=")
        expected_result = ["202205140500 TAF LRCL 140500Z 1406/1415 VRB04KT CAVOK", "PROB30 1312/1317 VRB18G28KT",
                           "PROB40 TEMPO 1407/1409 33012KT",
                           "PROB30 BECMG 1407/1409 33012KT"]
        self.assertEqual(actual_result, expected_result)

    def testInvalidFormatting(self):
        actual_result = taf_parser.parse_code(
            "202309241100 TAF LRCL 241100Z 2412/2421 VRB04KT 9999 SCT050                       TEMPO 2412/2415 "
            "VRB15G25KT 5000 TSRA BKN010                        BKN030CB                       BECMG 2413/2415 "
            "30010KT"
        )
        expected_result = ["202309241100 TAF LRCL 241100Z 2412/2421 VRB04KT 9999 SCT050",
                           "TEMPO 2412/2415 VRB15G25KT 5000 TSRA BKN010 BKN030CB", "BECMG 2413/2415 30010KT"]
        self.assertEqual(actual_result, expected_result)

    def testParseTafCavok(self):
        actual_result = taf_parser.parse_taf("202205130200 TAF LRCL 130200Z 1303/1312 VRB04KT CAVOK")
        expected_result = [datetime.datetime(2022, 5, 13, 2, 0), "TAF", datetime.datetime(2022, 5, 13, 3, 0),
                           datetime.datetime(2022, 5, 13, 12, 0), None, 4, True, None, True, 9999] + 5 * [None] + [
                              False] + 2 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testParseTafCorCavok(self):
        actual_result = taf_parser.parse_taf("202205130200 TAF COR LRCL 130200Z 1303/1312 VRB04KT CAVOK")
        expected_result = [datetime.datetime(2022, 5, 13, 2, 0), "TAF COR", datetime.datetime(2022, 5, 13, 3, 0),
                           datetime.datetime(2022, 5, 13, 12, 0), None, 4, True, None, True, 9999] + 5 * [None] + [
                              False] + 2 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testParseTafAmdCavok(self):
        actual_result = taf_parser.parse_taf("202205130200 TAF AMD LRCL 130200Z 1303/1312 VRB04KT CAVOK")
        expected_result = [datetime.datetime(2022, 5, 13, 2, 0), "TAF AMD", datetime.datetime(2022, 5, 13, 3, 0),
                           datetime.datetime(2022, 5, 13, 12, 0), None, 4, True, None, True, 9999] + 5 * [None] + [
                              False] + 2 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testParseTafGust(self):
        actual_result = taf_parser.parse_taf("202205130200 TAF AMD LRCL 130200Z 1303/1312 22012G24KT CAVOK")
        expected_result = [datetime.datetime(2022, 5, 13, 2, 0), "TAF AMD", datetime.datetime(2022, 5, 13, 3, 0),
                           datetime.datetime(2022, 5, 13, 12, 0), 220, 12, False, 24, True, 9999, None] + 4 * [None] + [
                              False] + 2 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testWindVariabilityWithGust(self):
        actual_result = taf_parser.parse_taf("202205130200 TAF AMD LRCL 130200Z 1303/1312 VRB12G24KT CAVOK")
        expected_result = [datetime.datetime(2022, 5, 13, 2, 0), "TAF AMD", datetime.datetime(2022, 5, 13, 3, 0),
                           datetime.datetime(2022, 5, 13, 12, 0), None, 12, True, 24, True, 9999, None] + 4 * [None] + [
                              False] + 2 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testOnePresentPhenomena(self):
        actual_result = taf_parser.parse_taf("202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", None,
                           None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testTwoPresentPhenomena(self):
        actual_result = taf_parser.parse_taf("202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG +RAFG")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", "+RAFG",
                           None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testThreePresentPhenomena(self):
        actual_result = taf_parser.parse_taf("202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG +RAFG BCFG")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", "+RAFG",
                           "BCFG"] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testOneLayerOfClouds(self):
        actual_result = taf_parser.parse_taf(
            "202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG +RAFG BCFG SCT045")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", "+RAFG",
                           "BCFG", "SCT", 45, False] + 2 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testOneLayerOfCloudsCB(self):
        actual_result = taf_parser.parse_taf(
            "202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG +RAFG BCFG SCT045CB")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", "+RAFG",
                           "BCFG", "SCT", 45, True] + 2 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testTwoLayersOfClouds(self):
        actual_result = taf_parser.parse_taf(
            "202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG +RAFG BCFG SCT045CB FEW135")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", "+RAFG",
                           "BCFG", "SCT", 45, True, "FEW", 135, False] + [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testThreeLayersOfClouds(self):
        actual_result = taf_parser.parse_taf(
            "202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG +RAFG BCFG SCT045CB FEW135 BKN001")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", "+RAFG",
                           "BCFG", "SCT", 45, True, "FEW", 135, False] + ["BKN", 1, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testThreeLayersOfCloudsCB(self):
        actual_result = taf_parser.parse_taf(
            "202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG +RAFG BCFG SCT045CB FEW135CB BKN001CB")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", "+RAFG",
                           "BCFG", "SCT", 45, True, "FEW", 135, True] + ["BKN", 1, True] + [None]
        self.assertEqual(expected_result, actual_result)

    def testVerticalVisibility(self):
        actual_result = taf_parser.parse_taf(
            "202309210500 TAF LRCL 210500Z 2106/2115 VRB04KT 0200 FG +RAFG BCFG SCT045CB FEW135CB BKN001CB VV001")
        expected_result = [datetime.datetime(2023, 9, 21, 5, 0), "TAF", datetime.datetime(2023, 9, 21, 6, 0),
                           datetime.datetime(2023, 9, 21, 15, 0), None, 4, True, None, False, 200, "FG", "+RAFG",
                           "BCFG", "SCT", 45, True, "FEW", 135, True, "BKN", 1, True, 1]
        self.assertEqual(expected_result, actual_result)

    def testProbCavok(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2022, 2, 18, 6), "BECMG 1807/1809 13010KT CAVOK")
        expected_result = [None, "BECMG", datetime.datetime(2022, 2, 18, 7), datetime.datetime(2022, 2, 18, 9), 130, 10,
                           False, None, True, 9999] + 3 * [None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testProbGust(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2022, 2, 18, 6), "BECMG 1807/1809 22012G24KT CAVOK")
        expected_result = [None, "BECMG", datetime.datetime(2022, 2, 18, 7), datetime.datetime(2022, 2, 18, 9), 220, 12,
                           False, 24, True, 9999] + 3 * [None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testProbWindVariability(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2022, 2, 18, 6), "BECMG 1807/1809 VRB04KT CAVOK")
        expected_result = [None, "BECMG", datetime.datetime(2022, 2, 18, 7), datetime.datetime(2022, 2, 18, 9), None, 4,
                           True, None, True, 9999] + 3 * [None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testBECMGProb(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2022, 2, 18, 6),
                                                  "PROB40 BECMG 1807/1809 VRB04KT CAVOK")
        expected_result = [40, "BECMG", datetime.datetime(2022, 2, 18, 7), datetime.datetime(2022, 2, 18, 9), None, 4,
                           True, None, True, 9999] + 3 * [None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testProb30(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2022, 2, 18, 6),
                                                  "PROB30 1807/1809 VRB04KT CAVOK")
        expected_result = [30, "PROB", datetime.datetime(2022, 2, 18, 7), datetime.datetime(2022, 2, 18, 9), None, 4,
                           True, None, True, 9999] + 3 * [None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testProbWindVariabilityWithGust(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2022, 2, 18, 6),
                                                  "PROB30 1807/1809 VRB12G24KT CAVOK")
        expected_result = [30, "PROB", datetime.datetime(2022, 2, 18, 7), datetime.datetime(2022, 2, 18, 9), None, 12,
                           True, 24, True, 9999] + 3 * [None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testProbOnePresentPhenomena(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", None, None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testProbTwoPresentPhenomena(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG +RAFG")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", "+RAFG", None] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testProbThreePresentPhenomena(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG +RAFG BCFG")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", "+RAFG", "BCFG"] + 3 * [None, None, False] + [None]
        self.assertEqual(expected_result, actual_result)

    def testProbOneLayerOfClouds(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG +RAFG BCFG SCT045")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", "+RAFG", "BCFG", "SCT", 45, False] + 2 * [None, None, False] + [
                              None]
        self.assertEqual(expected_result, actual_result)

    def testProbOneLayerOfCloudsCB(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG +RAFG BCFG SCT045CB")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", "+RAFG", "BCFG", "SCT", 45, True] + 2 * [None, None, False] + [
                              None]
        self.assertEqual(expected_result, actual_result)

    def testProbTwoLayersOfClouds(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG +RAFG BCFG SCT045CB FEW135")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", "+RAFG", "BCFG", "SCT", 45, True, "FEW", 135, False] + [None,
                                                                                                               None,
                                                                                                               False] + [
                              None]
        self.assertEqual(expected_result, actual_result)

    def testProbThreeLayersOfClouds(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG +RAFG BCFG SCT045CB FEW135 "
                                                  "BKN001")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", "+RAFG", "BCFG", "SCT", 45, True, "FEW", 135, False, "BKN",
                           1, False, None]
        self.assertEqual(expected_result, actual_result)

    def testProbThreeLayersOfCloudsCB(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG +RAFG BCFG SCT045CB FEW135CB "
                                                  "BKN001CB")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", "+RAFG", "BCFG", "SCT", 45, True, "FEW", 135, True, "BKN",
                           1, True, None]
        self.assertEqual(expected_result, actual_result)

    def testProbVerticalVisibility(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2023, 9, 18, 7, 0),
                                                  "PROB30 1807/1809 VRB12G24KT 0200 FG +RAFG BCFG SCT045CB FEW135CB "
                                                  "BKN001CB VV001")
        expected_result = [30, "PROB", datetime.datetime(2023, 9, 18, 7), datetime.datetime(2023, 9, 18, 9), None, 12,
                           True, 24, False, 200, "FG", "+RAFG", "BCFG", "SCT", 45, True, "FEW", 135, True, "BKN",
                           1, True, 1]
        self.assertEqual(expected_result, actual_result)

    def testProb30Tempo(self):
        actual_result = taf_parser.parse_taf_prob(datetime.datetime(2021, 9, 12, 17, 0),
                                                  "PROB30 TEMPO 1223/1303 5000 BR")
        expected_result = [30, "TEMPO", datetime.datetime(2021, 9, 12, 23), datetime.datetime(2021, 9, 13, 3), None,
                           None,
                           False, None, False, 5000, "BR", None, None, None, None, False, None, None, False, None,
                           None, False, None]
        self.assertEqual(expected_result, actual_result)
