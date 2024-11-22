import datetime
import unittest
import metar_parser


class MetarParserTest(unittest.TestCase):

    def testCavok(self):
        actual_result = metar_parser.parse_metar("202111161000 METAR LRCL 161000Z 13009KT CAVOK 09/03 Q1028=")
        expected_result = [datetime.datetime(2021, 11, 16, 10, 00), "METAR", None, 130, 9, False, None, None, None,
                           True, 9999] + 24 * [None] + [9, 3, 1028]
        self.assertEqual(actual_result, expected_result)

    def testWindDirectionVariability(self):
        actual_result = metar_parser.parse_metar("202111251230 METAR LRCL 251230Z 07005KT 030V100 CAVOK 04/00 Q1017=")
        expected_result = [datetime.datetime(2021, 11, 25, 12, 30), "METAR", None, 70, 5, False, None, 30, 100, True,
                           9999] + 24 * [None] + [4, 0, 1017]
        self.assertEqual(actual_result, expected_result)

    def testOneLayerOfCloud(self):
        actual_result = metar_parser.parse_metar("202111251200 METAR LRCL 251200Z 07005KT 9999 FEW003 03/01 Q1017=")
        expected_result = [datetime.datetime(2021, 11, 25, 12, 00), "METAR", None, 70, 5, False, None, None, None,
                           False, 9999] + 13 * [None] + ["FEW", 3, None] + 8 * [None] + [3, 1, 1017]
        self.assertEqual(actual_result, expected_result)

    def testCorCallsign(self):
        actual_result = metar_parser.parse_metar("202111251200 METAR COR LRCL 251200Z 07005KT 9999 FEW003 03/01 Q1017=")
        expected_result = [datetime.datetime(2021, 11, 25, 12, 00), "METAR COR", None, 70, 5, False, None, None, None,
                           False, 9999] + 13 * [None] + ["FEW", 3, None] + 8 * [None] + [3, 1, 1017]
        self.assertEqual(actual_result, expected_result)

    def testSpeci(self):
        actual_result = metar_parser.parse_metar("202111251200 SPECI LRCL 251200Z 07005KT 9999 FEW003 03/01 Q1017=")
        expected_result = [datetime.datetime(2021, 11, 25, 12, 00), "SPECI", None, 70, 5, False, None, None,
                           None,
                           False, 9999] + 13 * [None] + ["FEW", 3, None] + 8 * [None] + [3, 1, 1017]
        self.assertEqual(actual_result, expected_result)

    def testSpeciCor(self):
        actual_result = metar_parser.parse_metar("202111251200 SPECI COR LRCL 251200Z 07005KT 9999 FEW003 03/01 Q1017=")
        expected_result = [datetime.datetime(2021, 11, 25, 12, 00), "SPECI COR", None, 70, 5, False, None, None,
                           None, False, 9999] + 13 * [None] + ["FEW", 3, None] + 8 * [None] + [3, 1, 1017]
        self.assertEqual(actual_result, expected_result)

    def testNilCallsign(self):
        actual_result = metar_parser.parse_metar("202111251200 METAR LRCL 251200Z NIL=")
        expected_result = [datetime.datetime(2021, 11, 25, 12, 00), "METAR", "NIL", None] + [None] * 34
        self.assertEqual(actual_result, expected_result)

    def testAutoCallsign(self):
        actual_result = metar_parser.parse_metar(
            "202111251200 METAR LRCL 251200Z AUTO 07005KT 9999 FEW003 03/01 Q1017=")
        expected_result = [datetime.datetime(2021, 11, 25, 12, 00), "METAR", "AUTO", 70, 5, False, None, None,
                           None, False, 9999] + 13 * [None] + ["FEW", 3, None] + 8 * [None] + [3, 1, 1017]
        self.assertEqual(actual_result, expected_result)

    def testWindVariability(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB01KT CAVOK M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 1, True, None, 60, 180,
                           True,
                           9999] + 24 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testGust(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z 22012G24KT CAVOK M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, 220, 12, False, 24, None, None,
                           True,
                           9999] + 24 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testWindVariabilityWithGust(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT CAVOK M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           True,
                           9999] + 24 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibility(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/1300 M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, 1300] + 20 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibilityTendency(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/1300N M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, 1300, None, 'N'] + 18 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibilityVariableDistanceDecreasing(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/1300VD M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, 1300, None, 'D'] + 18 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibilityIndicator(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1300 M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, 1300, 'M', None] + 18 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibilityVariability(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/1800V2000 M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, None, None, None, 1800, 2000] + 16 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibilityVariabilityTendency(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/1800V2000U M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, None, None, 'U', 1800, 2000] + 16 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibilityVariabilityMinimalIndicator(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800V2000U M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, None, None, 'U', 1800, 2000, 'M'] + 15 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibilityVariabilityMaximalIndicator(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/1800VP2000U M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, None, None, 'U', 1800, 2000, None, 'P'] + 14 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testRunwayVisibilityVariabilityIndicators(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P'] + 14 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testOnePhenomenon(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG"] + 13 * [None] + [0, -2,
                                                                                                                 1025]
        self.assertEqual(actual_result, expected_result)

    def testTwoPhenomena(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False,
                           1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA"] + 12 * [
                              None] + [
                              0, -2,
                              1025]
        self.assertEqual(actual_result, expected_result)

    def testThreePhenomena(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA"] + 11 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testThreePhenomenaNSC(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA NSC M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA"] + 9 * [None] + ["NSC", None, 0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testThreePhenomenaNCD(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA NCD M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA"] + 9 * [None] + ["NCD", None, 0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testOneCloudLayer(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA BKN001 M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA", "BKN", 1, None] + 8 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testOneCloudLayerCB(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA BKN001CB M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA", "BKN", 1, "CB"] + 8 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testOneCloudLayerTCU(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA BKN001TCU M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA", "BKN", 1, "TCU"] + 8 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testOneCloudLayerTypeNotDetected(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA BKN001/// M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA", "BKN", 1, "///"] + 8 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testOneCloudLayerCBNoNebulosityNoAltitude(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA //////CB M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA", None, None, "CB"] + 8 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testOneCloudLayerTCUNoNebulosityNoAltitude(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA //////TCU M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA", None, None, "TCU"] + 8 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testTwoCloudLayers(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA //////TCU FEW013 M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA", None, None, "TCU", "FEW", 13, None] + 5 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testThreeCloudLayers(self):
        actual_result = metar_parser.parse_metar(
            "202111231900 METAR COR LRCL 231900Z VRB08G18KT 1000 R25/M1800VP2000U BCFG +RA VCRA //////TCU FEW013 OVC060/// M00/M02 Q1025=")
        expected_result = [datetime.datetime(2021, 11, 23, 19, 00), "METAR COR", None, None, 8, True, 18, 60, 180,
                           False, 1000, None, None, 25, None, None, 'U', 1800, 2000, 'M', 'P', "BCFG", "+RA",
                           "VCRA", None, None, "TCU", "FEW", 13, None, "OVC", 60, "///"] + 2 * [None] + [0, -2, 1025]
        self.assertEqual(actual_result, expected_result)

    def testDirectionalVisibilityOnePhenomenon(self):
        actual_result = metar_parser.parse_metar(
            "202302250430 METAR LRCL 250430Z 26004KT 5000 2000E BR BKN047 02/02 Q1000=")
        expected_result = [datetime.datetime(2023, 2, 25, 4, 30), "METAR", None, 260, 4, False, None, None, None, False,
                           5000, 2000, 'E'] + 8 * [None] + ["BR", None, None, "BKN", 47] + 9 * [None] + [2, 2, 1000]
        self.assertEqual(actual_result, expected_result)
