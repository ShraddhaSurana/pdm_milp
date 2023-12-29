\* work_package_assay_scheduling *\
Minimize
_total_delay: _0_delay + _1_delay + _2_delay + _3_delay + _4_delay + _5_delay
 + _6_delay
Subject To
0_1_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_1_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__1_R2D2_(10041062_CAD) - _end_0_4 + _schedulestart_1_2
 >= -3000000
0_1_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_1_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__1_R2D2_(10041062_CAD) - _end_1_4 + _schedulestart_0_2
 >= -2000000
0_1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_0_4
 + _schedulestart_1_2 >= -3000000
0_1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_1_4
 + _schedulestart_0_2 >= -2000000
0_1_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__ - _end_0_4
 + _schedulestart_1_2 >= -3000000
0_1_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _0_1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__ - _end_1_4
 + _schedulestart_0_2 >= -2000000
0_2_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_2_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__2_R2D2_(10041062_CAD) - _end_0_4 + _schedulestart_2_2
 >= -3000000
0_2_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_2_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__2_R2D2_(10041062_CAD) - _end_2_4 + _schedulestart_0_2
 >= -2000000
0_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_0_4
 + _schedulestart_2_2 >= -3000000
0_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_2_4
 + _schedulestart_0_2 >= -2000000
0_2_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__ - _end_0_4
 + _schedulestart_2_2 >= -3000000
0_2_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _0_2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__ - _end_2_4
 + _schedulestart_0_2 >= -2000000
0_3_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_3_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__3_R2D2_(10041062_CAD) - _end_0_4 + _schedulestart_3_2
 >= -3000000
0_3_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_3_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__3_R2D2_(10041062_CAD) - _end_3_4 + _schedulestart_0_2
 >= -2000000
0_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_0_4
 + _schedulestart_3_2 >= -3000000
0_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_3_4
 + _schedulestart_0_2 >= -2000000
0_3_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__ - _end_0_4
 + _schedulestart_3_2 >= -3000000
0_3_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _0_3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__ - _end_3_4
 + _schedulestart_0_2 >= -2000000
0_4_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_4_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__4_R2D2_(10041062_CAD) - _end_0_4 + _schedulestart_4_2
 >= -3000000
0_4_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_4_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__4_R2D2_(10041062_CAD) - _end_4_4 + _schedulestart_0_2
 >= -2000000
0_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_0_4
 + _schedulestart_4_2 >= -3000000
0_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_4_4
 + _schedulestart_0_2 >= -2000000
0_4_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__ - _end_0_4
 + _schedulestart_4_2 >= -3000000
0_4_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _0_4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__ - _end_4_4
 + _schedulestart_0_2 >= -2000000
0_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_5_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_0_4 + _schedulestart_5_2
 >= -3000000
0_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_5_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_5_4 + _schedulestart_0_2
 >= -2000000
0_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_0_4
 + _schedulestart_5_2 >= -3000000
0_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_5_4
 + _schedulestart_0_2 >= -2000000
0_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_0_4
 + _schedulestart_5_2 >= -3000000
0_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _0_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_5_4
 + _schedulestart_0_2 >= -2000000
0_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_6_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_0_4 + _schedulestart_6_2
 >= -3000000
0_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_6_R2D2_(10041062_CAD) - 1000000 _asset__0_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_6_4 + _schedulestart_0_2
 >= -2000000
0_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_0_4
 + _schedulestart_6_2 >= -3000000
0_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _0_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_6_4
 + _schedulestart_0_2 >= -2000000
0_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _0_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_0_4
 + _schedulestart_6_2 >= -3000000
0_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _0_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__0_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_6_4
 + _schedulestart_0_2 >= -2000000
0_end_time_step2_less_than_end_of_start_time_of_shift1: - 480 _0_shift_start_1
 + _end_0_2 <= 480
0_shift_start_is_day_start_*_3: - 3 _0_day_start_0 + _0_shift_start_1 = 0
0_start_time_is_greater_than_shift_start_time: - 480 _0_shift_start_1
 + _schedulestart_0_1 >= 0
0_step4_start_time_and_end_time_is_in_same_shift: - 480 _0_shift_start_4
 + _schedulestart_0_4 >= 0
0_step5_start_time_and_end_time_is_in_same_shift: - 480 _0_shift_start_5
 + _end_0_5 <= 480
1_2_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_2_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__2_R2D2_(10041062_CAD) - _end_1_4 + _schedulestart_2_2
 >= -3000000
1_2_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_2_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__2_R2D2_(10041062_CAD) - _end_2_4 + _schedulestart_1_2
 >= -2000000
1_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_1_4
 + _schedulestart_2_2 >= -3000000
1_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_2_4
 + _schedulestart_1_2 >= -2000000
1_2_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__ - _end_1_4
 + _schedulestart_2_2 >= -3000000
1_2_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _1_2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__ - _end_2_4
 + _schedulestart_1_2 >= -2000000
1_3_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_3_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__3_R2D2_(10041062_CAD) - _end_1_4 + _schedulestart_3_2
 >= -3000000
1_3_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_3_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__3_R2D2_(10041062_CAD) - _end_3_4 + _schedulestart_1_2
 >= -2000000
1_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_1_4
 + _schedulestart_3_2 >= -3000000
1_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_3_4
 + _schedulestart_1_2 >= -2000000
1_3_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__ - _end_1_4
 + _schedulestart_3_2 >= -3000000
1_3_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _1_3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__ - _end_3_4
 + _schedulestart_1_2 >= -2000000
1_4_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_4_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__4_R2D2_(10041062_CAD) - _end_1_4 + _schedulestart_4_2
 >= -3000000
1_4_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_4_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__4_R2D2_(10041062_CAD) - _end_4_4 + _schedulestart_1_2
 >= -2000000
1_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_1_4
 + _schedulestart_4_2 >= -3000000
1_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_4_4
 + _schedulestart_1_2 >= -2000000
1_4_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__ - _end_1_4
 + _schedulestart_4_2 >= -3000000
1_4_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _1_4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__ - _end_4_4
 + _schedulestart_1_2 >= -2000000
1_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_5_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_1_4 + _schedulestart_5_2
 >= -3000000
1_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_5_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_5_4 + _schedulestart_1_2
 >= -2000000
1_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_1_4
 + _schedulestart_5_2 >= -3000000
1_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_5_4
 + _schedulestart_1_2 >= -2000000
1_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_1_4
 + _schedulestart_5_2 >= -3000000
1_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _1_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_5_4
 + _schedulestart_1_2 >= -2000000
1_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_6_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_1_4 + _schedulestart_6_2
 >= -3000000
1_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_6_R2D2_(10041062_CAD) - 1000000 _asset__1_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_6_4 + _schedulestart_1_2
 >= -2000000
1_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_1_4
 + _schedulestart_6_2 >= -3000000
1_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _1_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_6_4
 + _schedulestart_1_2 >= -2000000
1_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _1_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_1_4
 + _schedulestart_6_2 >= -3000000
1_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _1_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__1_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_6_4
 + _schedulestart_1_2 >= -2000000
1_end_time_step2_less_than_end_of_start_time_of_shift1: - 480 _1_shift_start_1
 + _end_1_2 <= 480
1_shift_start_is_day_start_*_3: - 3 _1_day_start_1 + _1_shift_start_1 = 0
1_start_time_is_greater_than_shift_start_time: - 480 _1_shift_start_1
 + _schedulestart_1_1 >= 0
1_step4_start_time_and_end_time_is_in_same_shift: - 480 _1_shift_start_4
 + _schedulestart_1_4 >= 0
1_step5_start_time_and_end_time_is_in_same_shift: - 480 _1_shift_start_5
 + _end_1_5 <= 480
2_3_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_3_R2D2_(10041062_CAD) - 1000000 _asset__2_R2D2_(10041062_CAD)
 - 1000000 _asset__3_R2D2_(10041062_CAD) - _end_2_4 + _schedulestart_3_2
 >= -3000000
2_3_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _2_3_R2D2_(10041062_CAD) - 1000000 _asset__2_R2D2_(10041062_CAD)
 - 1000000 _asset__3_R2D2_(10041062_CAD) - _end_3_4 + _schedulestart_2_2
 >= -2000000
2_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_2_4
 + _schedulestart_3_2 >= -3000000
2_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _2_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_3_4
 + _schedulestart_2_2 >= -2000000
2_3_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__ - _end_2_4
 + _schedulestart_3_2 >= -3000000
2_3_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _2_3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__ - _end_3_4
 + _schedulestart_2_2 >= -2000000
2_4_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_4_R2D2_(10041062_CAD) - 1000000 _asset__2_R2D2_(10041062_CAD)
 - 1000000 _asset__4_R2D2_(10041062_CAD) - _end_2_4 + _schedulestart_4_2
 >= -3000000
2_4_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _2_4_R2D2_(10041062_CAD) - 1000000 _asset__2_R2D2_(10041062_CAD)
 - 1000000 _asset__4_R2D2_(10041062_CAD) - _end_4_4 + _schedulestart_2_2
 >= -2000000
2_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_2_4
 + _schedulestart_4_2 >= -3000000
2_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _2_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_4_4
 + _schedulestart_2_2 >= -2000000
2_4_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__ - _end_2_4
 + _schedulestart_4_2 >= -3000000
2_4_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _2_4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__ - _end_4_4
 + _schedulestart_2_2 >= -2000000
2_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_5_R2D2_(10041062_CAD) - 1000000 _asset__2_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_2_4 + _schedulestart_5_2
 >= -3000000
2_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _2_5_R2D2_(10041062_CAD) - 1000000 _asset__2_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_5_4 + _schedulestart_2_2
 >= -2000000
2_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_2_4
 + _schedulestart_5_2 >= -3000000
2_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _2_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_5_4
 + _schedulestart_2_2 >= -2000000
2_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_2_4
 + _schedulestart_5_2 >= -3000000
2_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _2_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_5_4
 + _schedulestart_2_2 >= -2000000
2_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_6_R2D2_(10041062_CAD) - 1000000 _asset__2_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_2_4 + _schedulestart_6_2
 >= -3000000
2_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _2_6_R2D2_(10041062_CAD) - 1000000 _asset__2_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_6_4 + _schedulestart_2_2
 >= -2000000
2_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_2_4
 + _schedulestart_6_2 >= -3000000
2_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _2_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_6_4
 + _schedulestart_2_2 >= -2000000
2_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _2_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_2_4
 + _schedulestart_6_2 >= -3000000
2_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _2_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__2_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_6_4
 + _schedulestart_2_2 >= -2000000
2_end_time_step2_less_than_end_of_start_time_of_shift1: - 480 _2_shift_start_1
 + _end_2_2 <= 480
2_shift_start_is_day_start_*_3: - 3 _2_day_start_2 + _2_shift_start_1 = 0
2_start_time_is_greater_than_shift_start_time: - 480 _2_shift_start_1
 + _schedulestart_2_1 >= 0
2_step4_start_time_and_end_time_is_in_same_shift: - 480 _2_shift_start_4
 + _schedulestart_2_4 >= 0
2_step5_start_time_and_end_time_is_in_same_shift: - 480 _2_shift_start_5
 + _end_2_5 <= 480
3_4_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_4_R2D2_(10041062_CAD) - 1000000 _asset__3_R2D2_(10041062_CAD)
 - 1000000 _asset__4_R2D2_(10041062_CAD) - _end_3_4 + _schedulestart_4_2
 >= -3000000
3_4_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _3_4_R2D2_(10041062_CAD) - 1000000 _asset__3_R2D2_(10041062_CAD)
 - 1000000 _asset__4_R2D2_(10041062_CAD) - _end_4_4 + _schedulestart_3_2
 >= -2000000
3_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_3_4
 + _schedulestart_4_2 >= -3000000
3_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _3_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_4_4
 + _schedulestart_3_2 >= -2000000
3_4_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__ - _end_3_4
 + _schedulestart_4_2 >= -3000000
3_4_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _3_4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__ - _end_4_4
 + _schedulestart_3_2 >= -2000000
3_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_5_R2D2_(10041062_CAD) - 1000000 _asset__3_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_3_4 + _schedulestart_5_2
 >= -3000000
3_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _3_5_R2D2_(10041062_CAD) - 1000000 _asset__3_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_5_4 + _schedulestart_3_2
 >= -2000000
3_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_3_4
 + _schedulestart_5_2 >= -3000000
3_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _3_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_5_4
 + _schedulestart_3_2 >= -2000000
3_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_3_4
 + _schedulestart_5_2 >= -3000000
3_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _3_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_5_4
 + _schedulestart_3_2 >= -2000000
3_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_6_R2D2_(10041062_CAD) - 1000000 _asset__3_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_3_4 + _schedulestart_6_2
 >= -3000000
3_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _3_6_R2D2_(10041062_CAD) - 1000000 _asset__3_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_6_4 + _schedulestart_3_2
 >= -2000000
3_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_3_4
 + _schedulestart_6_2 >= -3000000
3_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _3_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_6_4
 + _schedulestart_3_2 >= -2000000
3_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _3_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_3_4
 + _schedulestart_6_2 >= -3000000
3_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _3_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__3_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_6_4
 + _schedulestart_3_2 >= -2000000
3_end_time_step2_less_than_end_of_start_time_of_shift1: - 480 _3_shift_start_1
 + _end_3_2 <= 480
3_shift_start_is_day_start_*_3: - 3 _3_day_start_3 + _3_shift_start_1 = 0
3_start_time_is_greater_than_shift_start_time: - 480 _3_shift_start_1
 + _schedulestart_3_1 >= 0
3_step4_start_time_and_end_time_is_in_same_shift: - 480 _3_shift_start_4
 + _schedulestart_3_4 >= 0
3_step5_start_time_and_end_time_is_in_same_shift: - 480 _3_shift_start_5
 + _end_3_5 <= 480
4_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _4_5_R2D2_(10041062_CAD) - 1000000 _asset__4_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_4_4 + _schedulestart_5_2
 >= -3000000
4_5_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _4_5_R2D2_(10041062_CAD) - 1000000 _asset__4_R2D2_(10041062_CAD)
 - 1000000 _asset__5_R2D2_(10041062_CAD) - _end_5_4 + _schedulestart_4_2
 >= -2000000
4_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _4_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_4_4
 + _schedulestart_5_2 >= -3000000
4_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _4_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_5_4
 + _schedulestart_4_2 >= -2000000
4_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _4_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_4_4
 + _schedulestart_5_2 >= -3000000
4_5_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _4_5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__ - _end_5_4
 + _schedulestart_4_2 >= -2000000
4_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _4_6_R2D2_(10041062_CAD) - 1000000 _asset__4_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_4_4 + _schedulestart_6_2
 >= -3000000
4_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _4_6_R2D2_(10041062_CAD) - 1000000 _asset__4_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_6_4 + _schedulestart_4_2
 >= -2000000
4_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _4_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_4_4
 + _schedulestart_6_2 >= -3000000
4_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _4_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_6_4
 + _schedulestart_4_2 >= -2000000
4_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _4_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_4_4
 + _schedulestart_6_2 >= -3000000
4_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _4_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__4_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_6_4
 + _schedulestart_4_2 >= -2000000
4_end_time_step2_less_than_end_of_start_time_of_shift1: - 480 _4_shift_start_1
 + _end_4_2 <= 480
4_shift_start_is_day_start_*_3: - 3 _4_day_start_4 + _4_shift_start_1 = 0
4_start_time_is_greater_than_shift_start_time: - 480 _4_shift_start_1
 + _schedulestart_4_1 >= 0
4_step4_start_time_and_end_time_is_in_same_shift: - 480 _4_shift_start_4
 + _schedulestart_4_4 >= 0
4_step5_start_time_and_end_time_is_in_same_shift: - 480 _4_shift_start_5
 + _end_4_5 <= 480
5_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _5_6_R2D2_(10041062_CAD) - 1000000 _asset__5_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_5_4 + _schedulestart_6_2
 >= -3000000
5_6_R2D2_(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _5_6_R2D2_(10041062_CAD) - 1000000 _asset__5_R2D2_(10041062_CAD)
 - 1000000 _asset__6_R2D2_(10041062_CAD) - _end_6_4 + _schedulestart_5_2
 >= -2000000
5_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_After:
 - 1000000 _5_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_5_4
 + _schedulestart_6_2 >= -3000000
5_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)_same_asset_not_involved_in_two_tests_Before:
 1000000 _5_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) - _end_6_4
 + _schedulestart_5_2 >= -2000000
5_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_After:
 - 1000000 _5_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_5_4
 + _schedulestart_6_2 >= -3000000
5_6_R2D2_(CAD)__(10045347_LC)___same_asset_not_involved_in_two_tests_Before:
 1000000 _5_6_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__5_R2D2_(CAD)__(10045347_LC)__
 - 1000000 _asset__6_R2D2_(CAD)__(10045347_LC)__ - _end_6_4
 + _schedulestart_5_2 >= -2000000
5_end_time_step2_less_than_end_of_start_time_of_shift1: - 480 _5_shift_start_1
 + _end_5_2 <= 480
5_shift_start_is_day_start_*_3: - 3 _5_day_start_5 + _5_shift_start_1 = 0
5_start_time_is_greater_than_shift_start_time: - 480 _5_shift_start_1
 + _schedulestart_5_1 >= 0
5_step4_start_time_and_end_time_is_in_same_shift: - 480 _5_shift_start_4
 + _schedulestart_5_4 >= 0
5_step5_start_time_and_end_time_is_in_same_shift: - 480 _5_shift_start_5
 + _end_5_5 <= 480
6_end_time_step2_less_than_end_of_start_time_of_shift1: - 480 _6_shift_start_1
 + _end_6_2 <= 480
6_shift_start_is_day_start_*_3: - 3 _6_day_start_6 + _6_shift_start_1 = 0
6_start_time_is_greater_than_shift_start_time: - 480 _6_shift_start_1
 + _schedulestart_6_1 >= 0
6_step4_start_time_and_end_time_is_in_same_shift: - 480 _6_shift_start_4
 + _schedulestart_6_4 >= 0
6_step5_start_time_and_end_time_is_in_same_shift: - 480 _6_shift_start_5
 + _end_6_5 <= 480
_0_1_dur: _0_duration_1 = 155
_0_1_end_is_equal_to_start_and_duration: - _0_duration_1 + _end_0_1
 - _schedulestart_0_1 = 0
_0_2_dur: _0_duration_2 = 45
_0_2_end_is_equal_to_start_and_duration: - _0_duration_2 + _end_0_2
 - _schedulestart_0_2 = 0
_0_3_dur: _0_duration_3 = 12
_0_3_end_is_equal_to_start_and_duration: - _0_duration_3 + _end_0_3
 - _schedulestart_0_3 = 0
_0_4_dur: _0_duration_4 = 15
_0_4_end_is_equal_to_start_and_duration: - _0_duration_4 + _end_0_4
 - _schedulestart_0_4 = 0
_0_5_dur: _0_duration_5 = 65
_0_5_end_is_equal_to_start_and_duration: - _0_duration_5 + _end_0_5
 - _schedulestart_0_5 = 0
_0_Only_one_asset_can_be_assigned: _asset__0_R2D2_(10041062_CAD)
 + _asset__0_R2D2_(CAD)__(10045347_LC)__
 + _asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) = 1
_0_delay_time: _0_delay + _0_due_date - _end_0_5 = 0
_0_due_date_for_each_test: _0_due_date = 152640
_0_step2_should_start_immediately_after_step1: - _end_0_1 + _schedulestart_0_2
 = 0
_0_step3_should_start_immediately_after_step2: - _end_0_2 + _schedulestart_0_3
 = 0
_0_step4_should_start_after_step3: - _end_0_3 + _schedulestart_0_4 >= 0
_0_step5_should_start_after_step4: - _end_0_4 + _schedulestart_0_5 >= 0
_1_1_dur: _1_duration_1 = 155
_1_1_end_is_equal_to_start_and_duration: - _1_duration_1 + _end_1_1
 - _schedulestart_1_1 = 0
_1_2_dur: _1_duration_2 = 45
_1_2_end_is_equal_to_start_and_duration: - _1_duration_2 + _end_1_2
 - _schedulestart_1_2 = 0
_1_3_dur: _1_duration_3 = 12
_1_3_end_is_equal_to_start_and_duration: - _1_duration_3 + _end_1_3
 - _schedulestart_1_3 = 0
_1_4_dur: _1_duration_4 = 15
_1_4_end_is_equal_to_start_and_duration: - _1_duration_4 + _end_1_4
 - _schedulestart_1_4 = 0
_1_5_dur: _1_duration_5 = 65
_1_5_end_is_equal_to_start_and_duration: - _1_duration_5 + _end_1_5
 - _schedulestart_1_5 = 0
_1_Only_one_asset_can_be_assigned: _asset__1_R2D2_(10041062_CAD)
 + _asset__1_R2D2_(CAD)__(10045347_LC)__
 + _asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) = 1
_1_delay_time: _1_delay + _1_due_date - _end_1_5 = 0
_1_due_date_for_each_test: _1_due_date = 152640
_1_step2_should_start_immediately_after_step1: - _end_1_1 + _schedulestart_1_2
 = 0
_1_step3_should_start_immediately_after_step2: - _end_1_2 + _schedulestart_1_3
 = 0
_1_step4_should_start_after_step3: - _end_1_3 + _schedulestart_1_4 >= 0
_1_step5_should_start_after_step4: - _end_1_4 + _schedulestart_1_5 >= 0
_2_1_dur: _2_duration_1 = 155
_2_1_end_is_equal_to_start_and_duration: - _2_duration_1 + _end_2_1
 - _schedulestart_2_1 = 0
_2_2_dur: _2_duration_2 = 45
_2_2_end_is_equal_to_start_and_duration: - _2_duration_2 + _end_2_2
 - _schedulestart_2_2 = 0
_2_3_dur: _2_duration_3 = 12
_2_3_end_is_equal_to_start_and_duration: - _2_duration_3 + _end_2_3
 - _schedulestart_2_3 = 0
_2_4_dur: _2_duration_4 = 15
_2_4_end_is_equal_to_start_and_duration: - _2_duration_4 + _end_2_4
 - _schedulestart_2_4 = 0
_2_5_dur: _2_duration_5 = 65
_2_5_end_is_equal_to_start_and_duration: - _2_duration_5 + _end_2_5
 - _schedulestart_2_5 = 0
_2_Only_one_asset_can_be_assigned: _asset__2_R2D2_(10041062_CAD)
 + _asset__2_R2D2_(CAD)__(10045347_LC)__
 + _asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) = 1
_2_delay_time: _2_delay + _2_due_date - _end_2_5 = 0
_2_due_date_for_each_test: _2_due_date = 154080
_2_step2_should_start_immediately_after_step1: - _end_2_1 + _schedulestart_2_2
 = 0
_2_step3_should_start_immediately_after_step2: - _end_2_2 + _schedulestart_2_3
 = 0
_2_step4_should_start_after_step3: - _end_2_3 + _schedulestart_2_4 >= 0
_2_step5_should_start_after_step4: - _end_2_4 + _schedulestart_2_5 >= 0
_3_1_dur: _3_duration_1 = 155
_3_1_end_is_equal_to_start_and_duration: - _3_duration_1 + _end_3_1
 - _schedulestart_3_1 = 0
_3_2_dur: _3_duration_2 = 45
_3_2_end_is_equal_to_start_and_duration: - _3_duration_2 + _end_3_2
 - _schedulestart_3_2 = 0
_3_3_dur: _3_duration_3 = 12
_3_3_end_is_equal_to_start_and_duration: - _3_duration_3 + _end_3_3
 - _schedulestart_3_3 = 0
_3_4_dur: _3_duration_4 = 15
_3_4_end_is_equal_to_start_and_duration: - _3_duration_4 + _end_3_4
 - _schedulestart_3_4 = 0
_3_5_dur: _3_duration_5 = 65
_3_5_end_is_equal_to_start_and_duration: - _3_duration_5 + _end_3_5
 - _schedulestart_3_5 = 0
_3_Only_one_asset_can_be_assigned: _asset__3_R2D2_(10041062_CAD)
 + _asset__3_R2D2_(CAD)__(10045347_LC)__
 + _asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) = 1
_3_delay_time: _3_delay + _3_due_date - _end_3_5 = 0
_3_due_date_for_each_test: _3_due_date = 155520
_3_step2_should_start_immediately_after_step1: - _end_3_1 + _schedulestart_3_2
 = 0
_3_step3_should_start_immediately_after_step2: - _end_3_2 + _schedulestart_3_3
 = 0
_3_step4_should_start_after_step3: - _end_3_3 + _schedulestart_3_4 >= 0
_3_step5_should_start_after_step4: - _end_3_4 + _schedulestart_3_5 >= 0
_4_1_dur: _4_duration_1 = 155
_4_1_end_is_equal_to_start_and_duration: - _4_duration_1 + _end_4_1
 - _schedulestart_4_1 = 0
_4_2_dur: _4_duration_2 = 45
_4_2_end_is_equal_to_start_and_duration: - _4_duration_2 + _end_4_2
 - _schedulestart_4_2 = 0
_4_3_dur: _4_duration_3 = 12
_4_3_end_is_equal_to_start_and_duration: - _4_duration_3 + _end_4_3
 - _schedulestart_4_3 = 0
_4_4_dur: _4_duration_4 = 15
_4_4_end_is_equal_to_start_and_duration: - _4_duration_4 + _end_4_4
 - _schedulestart_4_4 = 0
_4_5_dur: _4_duration_5 = 65
_4_5_end_is_equal_to_start_and_duration: - _4_duration_5 + _end_4_5
 - _schedulestart_4_5 = 0
_4_Only_one_asset_can_be_assigned: _asset__4_R2D2_(10041062_CAD)
 + _asset__4_R2D2_(CAD)__(10045347_LC)__
 + _asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) = 1
_4_delay_time: _4_delay + _4_due_date - _end_4_5 = 0
_4_due_date_for_each_test: _4_due_date = 156960
_4_step2_should_start_immediately_after_step1: - _end_4_1 + _schedulestart_4_2
 = 0
_4_step3_should_start_immediately_after_step2: - _end_4_2 + _schedulestart_4_3
 = 0
_4_step4_should_start_after_step3: - _end_4_3 + _schedulestart_4_4 >= 0
_4_step5_should_start_after_step4: - _end_4_4 + _schedulestart_4_5 >= 0
_5_1_dur: _5_duration_1 = 155
_5_1_end_is_equal_to_start_and_duration: - _5_duration_1 + _end_5_1
 - _schedulestart_5_1 = 0
_5_2_dur: _5_duration_2 = 45
_5_2_end_is_equal_to_start_and_duration: - _5_duration_2 + _end_5_2
 - _schedulestart_5_2 = 0
_5_3_dur: _5_duration_3 = 12
_5_3_end_is_equal_to_start_and_duration: - _5_duration_3 + _end_5_3
 - _schedulestart_5_3 = 0
_5_4_dur: _5_duration_4 = 15
_5_4_end_is_equal_to_start_and_duration: - _5_duration_4 + _end_5_4
 - _schedulestart_5_4 = 0
_5_5_dur: _5_duration_5 = 65
_5_5_end_is_equal_to_start_and_duration: - _5_duration_5 + _end_5_5
 - _schedulestart_5_5 = 0
_5_Only_one_asset_can_be_assigned: _asset__5_R2D2_(10041062_CAD)
 + _asset__5_R2D2_(CAD)__(10045347_LC)__
 + _asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) = 1
_5_delay_time: _5_delay + _5_due_date - _end_5_5 = 0
_5_due_date_for_each_test: _5_due_date = 158400
_5_step2_should_start_immediately_after_step1: - _end_5_1 + _schedulestart_5_2
 = 0
_5_step3_should_start_immediately_after_step2: - _end_5_2 + _schedulestart_5_3
 = 0
_5_step4_should_start_after_step3: - _end_5_3 + _schedulestart_5_4 >= 0
_5_step5_should_start_after_step4: - _end_5_4 + _schedulestart_5_5 >= 0
_6_1_dur: _6_duration_1 = 155
_6_1_end_is_equal_to_start_and_duration: - _6_duration_1 + _end_6_1
 - _schedulestart_6_1 = 0
_6_2_dur: _6_duration_2 = 45
_6_2_end_is_equal_to_start_and_duration: - _6_duration_2 + _end_6_2
 - _schedulestart_6_2 = 0
_6_3_dur: _6_duration_3 = 12
_6_3_end_is_equal_to_start_and_duration: - _6_duration_3 + _end_6_3
 - _schedulestart_6_3 = 0
_6_4_dur: _6_duration_4 = 15
_6_4_end_is_equal_to_start_and_duration: - _6_duration_4 + _end_6_4
 - _schedulestart_6_4 = 0
_6_5_dur: _6_duration_5 = 65
_6_5_end_is_equal_to_start_and_duration: - _6_duration_5 + _end_6_5
 - _schedulestart_6_5 = 0
_6_Only_one_asset_can_be_assigned: _asset__6_R2D2_(10041062_CAD)
 + _asset__6_R2D2_(CAD)__(10045347_LC)__
 + _asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD) = 1
_6_delay_time: _6_delay + _6_due_date - _end_6_5 = 0
_6_due_date_for_each_test: _6_due_date = 158400
_6_step2_should_start_immediately_after_step1: - _end_6_1 + _schedulestart_6_2
 = 0
_6_step3_should_start_immediately_after_step2: - _end_6_2 + _schedulestart_6_3
 = 0
_6_step4_should_start_after_step3: - _end_6_3 + _schedulestart_6_4 >= 0
_6_step5_should_start_after_step4: - _end_6_4 + _schedulestart_6_5 >= 0
__schedulestart_0_1_available_date_for_each_test:
 __schedulestart_0_1_available_date = 100800
__schedulestart_0_1_cannot_start_before_avail:
 - __schedulestart_0_1_available_date + _schedulestart_0_1 >= 0
__schedulestart_1_1_available_date_for_each_test:
 __schedulestart_1_1_available_date = 100800
__schedulestart_1_1_cannot_start_before_avail:
 - __schedulestart_1_1_available_date + _schedulestart_1_1 >= 0
__schedulestart_2_1_available_date_for_each_test:
 __schedulestart_2_1_available_date = 100800
__schedulestart_2_1_cannot_start_before_avail:
 - __schedulestart_2_1_available_date + _schedulestart_2_1 >= 0
__schedulestart_3_1_available_date_for_each_test:
 __schedulestart_3_1_available_date = 100800
__schedulestart_3_1_cannot_start_before_avail:
 - __schedulestart_3_1_available_date + _schedulestart_3_1 >= 0
__schedulestart_4_1_available_date_for_each_test:
 __schedulestart_4_1_available_date = 100800
__schedulestart_4_1_cannot_start_before_avail:
 - __schedulestart_4_1_available_date + _schedulestart_4_1 >= 0
__schedulestart_5_1_available_date_for_each_test:
 __schedulestart_5_1_available_date = 100800
__schedulestart_5_1_cannot_start_before_avail:
 - __schedulestart_5_1_available_date + _schedulestart_5_1 >= 0
__schedulestart_6_1_available_date_for_each_test:
 __schedulestart_6_1_available_date = 100800
__schedulestart_6_1_cannot_start_before_avail:
 - __schedulestart_6_1_available_date + _schedulestart_6_1 >= 0
Bounds
 0 <= _0_day_start_0
 _0_delay free
 0 <= _0_shift_start_1
 0 <= _0_shift_start_4
 0 <= _0_shift_start_5
 0 <= _1_day_start_1
 _1_delay free
 0 <= _1_shift_start_1
 0 <= _1_shift_start_4
 0 <= _1_shift_start_5
 0 <= _2_day_start_2
 _2_delay free
 0 <= _2_shift_start_1
 0 <= _2_shift_start_4
 0 <= _2_shift_start_5
 0 <= _3_day_start_3
 _3_delay free
 0 <= _3_shift_start_1
 0 <= _3_shift_start_4
 0 <= _3_shift_start_5
 0 <= _4_day_start_4
 _4_delay free
 0 <= _4_shift_start_1
 0 <= _4_shift_start_4
 0 <= _4_shift_start_5
 0 <= _5_day_start_5
 _5_delay free
 0 <= _5_shift_start_1
 0 <= _5_shift_start_4
 0 <= _5_shift_start_5
 0 <= _6_day_start_6
 _6_delay free
 0 <= _6_shift_start_1
 0 <= _6_shift_start_4
 0 <= _6_shift_start_5
Generals
_0_day_start_0
_0_shift_start_1
_0_shift_start_4
_0_shift_start_5
_1_day_start_1
_1_shift_start_1
_1_shift_start_4
_1_shift_start_5
_2_day_start_2
_2_shift_start_1
_2_shift_start_4
_2_shift_start_5
_3_day_start_3
_3_shift_start_1
_3_shift_start_4
_3_shift_start_5
_4_day_start_4
_4_shift_start_1
_4_shift_start_4
_4_shift_start_5
_5_day_start_5
_5_shift_start_1
_5_shift_start_4
_5_shift_start_5
_6_day_start_6
_6_shift_start_1
_6_shift_start_4
_6_shift_start_5
Binaries
_0_1_R2D2_(10041062_CAD)
_0_1_R2D2_(CAD)__(10045347_LC)__
_0_1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_0_2_R2D2_(10041062_CAD)
_0_2_R2D2_(CAD)__(10045347_LC)__
_0_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_0_3_R2D2_(10041062_CAD)
_0_3_R2D2_(CAD)__(10045347_LC)__
_0_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_0_4_R2D2_(10041062_CAD)
_0_4_R2D2_(CAD)__(10045347_LC)__
_0_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_0_5_R2D2_(10041062_CAD)
_0_5_R2D2_(CAD)__(10045347_LC)__
_0_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_0_6_R2D2_(10041062_CAD)
_0_6_R2D2_(CAD)__(10045347_LC)__
_0_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_1_2_R2D2_(10041062_CAD)
_1_2_R2D2_(CAD)__(10045347_LC)__
_1_2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_1_3_R2D2_(10041062_CAD)
_1_3_R2D2_(CAD)__(10045347_LC)__
_1_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_1_4_R2D2_(10041062_CAD)
_1_4_R2D2_(CAD)__(10045347_LC)__
_1_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_1_5_R2D2_(10041062_CAD)
_1_5_R2D2_(CAD)__(10045347_LC)__
_1_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_1_6_R2D2_(10041062_CAD)
_1_6_R2D2_(CAD)__(10045347_LC)__
_1_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_2_3_R2D2_(10041062_CAD)
_2_3_R2D2_(CAD)__(10045347_LC)__
_2_3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_2_4_R2D2_(10041062_CAD)
_2_4_R2D2_(CAD)__(10045347_LC)__
_2_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_2_5_R2D2_(10041062_CAD)
_2_5_R2D2_(CAD)__(10045347_LC)__
_2_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_2_6_R2D2_(10041062_CAD)
_2_6_R2D2_(CAD)__(10045347_LC)__
_2_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_3_4_R2D2_(10041062_CAD)
_3_4_R2D2_(CAD)__(10045347_LC)__
_3_4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_3_5_R2D2_(10041062_CAD)
_3_5_R2D2_(CAD)__(10045347_LC)__
_3_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_3_6_R2D2_(10041062_CAD)
_3_6_R2D2_(CAD)__(10045347_LC)__
_3_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_4_5_R2D2_(10041062_CAD)
_4_5_R2D2_(CAD)__(10045347_LC)__
_4_5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_4_6_R2D2_(10041062_CAD)
_4_6_R2D2_(CAD)__(10045347_LC)__
_4_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_5_6_R2D2_(10041062_CAD)
_5_6_R2D2_(CAD)__(10045347_LC)__
_5_6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_asset__0_R2D2_(10041062_CAD)
_asset__0_R2D2_(CAD)__(10045347_LC)__
_asset__0_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_asset__1_R2D2_(10041062_CAD)
_asset__1_R2D2_(CAD)__(10045347_LC)__
_asset__1_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_asset__2_R2D2_(10041062_CAD)
_asset__2_R2D2_(CAD)__(10045347_LC)__
_asset__2_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_asset__3_R2D2_(10041062_CAD)
_asset__3_R2D2_(CAD)__(10045347_LC)__
_asset__3_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_asset__4_R2D2_(10041062_CAD)
_asset__4_R2D2_(CAD)__(10045347_LC)__
_asset__4_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_asset__5_R2D2_(10041062_CAD)
_asset__5_R2D2_(CAD)__(10045347_LC)__
_asset__5_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
_asset__6_R2D2_(10041062_CAD)
_asset__6_R2D2_(CAD)__(10045347_LC)__
_asset__6_R2D2_(CAD)__(10045347_LC)__(10041062_CAD)
End
