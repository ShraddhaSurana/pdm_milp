from datetime import datetime

import numpy as np
import pandas as pd
from pulp import LpVariable, LpProblem, LpMinimize, lpSum


def load_data(path):
    df = pd.read_csv(path)
    return df


if __name__ == '__main__':
    ##################################################################################################################
    # Cannot start before available day constraints
    df_assay_to_be_scheduled = load_data("data/assay_to_be_scheduled.csv")

    forecast_assay_list_step1 = []
    available_dates = []
    business_days_list = []
    unique_id = []

    pdm_scheduler = LpProblem(
        'work_package_assay_scheduling', LpMinimize)

    today_date = datetime.today().date()

    # looping through all the assays
    for row in range(0, len(df_assay_to_be_scheduled)):
        unique_id.append(str(row))

        start = "_start_" + str(row) + "_1" #step 1 *Tij
        var_start = LpVariable(start, 0, None)

        forecast_assay_list_step1.append(var_start)
        available_dates.append(pd.to_datetime(df_assay_to_be_scheduled.iloc[row]['Available date']).date())

        # treating time available as running minutes form.
        business_day_minutes = np.busday_count(today_date, pd.to_datetime(df_assay_to_be_scheduled.iloc[row]['Available date']).date()) * 1440

        if business_day_minutes < 0:
            business_days_list.append(0)
        else:
            business_days_list.append(business_day_minutes)

    df_assay_to_be_scheduled['ID'] = unique_id
    print(forecast_assay_list_step1)
    print(available_dates)
    print(business_days_list)

    # looping though step 1 of all the assays
    for index in range(0, len(forecast_assay_list_step1)):
        var_start = forecast_assay_list_step1[index] #start time of step 1 of every line item (assay)

        available_date = LpVariable(f'_{forecast_assay_list_step1[index]}_available_date')

        pdm_scheduler += (available_date == business_days_list[index]) #object of class lpProblem. Adding constraint that available date == business date

        pdm_scheduler += var_start >= available_date, f'_{forecast_assay_list_step1[index]}_cannot_start_before_avail' # last part is giving it a readable name

    ##################################################################################################################
    # Assay Constraints
    df_asset_assay_mapping = load_data("data/asset_assay_mapping.csv")
    asset_test_mapping = df_asset_assay_mapping.groupby("assay").agg({"assetid": list}).reset_index()
    asset_test_mapping.rename(columns={'assay': 'Assay'}, inplace=True)

    merged_assay_data = asset_test_mapping.merge(
        df_assay_to_be_scheduled,
        on="Assay",
        how="right",
    )
    assay_list_of_compatible_assets = []#new
    assay_asset_id = []
    for mapping in range(0, len(merged_assay_data)):
        asset_assay_list = []
        assets_for_each_assay = []#new
        for asset_index in merged_assay_data.iloc[mapping]['assetid']:
            asset_assay_list.append("asset_ " + merged_assay_data.iloc[mapping]['ID'] + "_" + asset_index)
            assets_for_each_assay.append(asset_index)#new
        assay_list_of_compatible_assets.append(assets_for_each_assay) #new

        assay_asset_id.append(asset_assay_list)

    merged_assay_data['assay_asset_id'] = assay_asset_id

    index = 0
    assay_list_of_compatible_assets_binary_variable = [] # new
    for assay_asset_id_list in merged_assay_data['assay_asset_id']:
        assay_asset_binary_variable_list = []
        for assay_asset_id in assay_asset_id_list:
            variable = LpVariable(f'_{assay_asset_id}', cat='Binary')
            assay_asset_binary_variable_list.append(variable)
        assay_list_of_compatible_assets_binary_variable.append(assay_asset_binary_variable_list) #new

        pdm_scheduler += lpSum(assay_asset_binary_variable_list) == 1, f"_{index}_Only one asset can be assigned"

        index += 1

    ##################################################################################################################
    df_assay_duration = load_data("data/assay_duration.csv")

    # for Step Constraints
    assay_name_list = []
    assay_id_list = []
    assay_step_duration_list = []
    for row in range(0, len(df_assay_duration)):
        assay = df_assay_duration.iloc[row]
        assay_name = assay.values[0]

        step_duration_list = []

        assay_name_list.append(assay_name)
        stepwise_duration = assay.values[1:]

        step_duration_list.append(sum(stepwise_duration[0:7]))
        step_duration_list.append(stepwise_duration[7])
        step_duration_list.append(stepwise_duration[8])
        step_duration_list.append(stepwise_duration[9])
        step_duration_list.append(sum(stepwise_duration[10:]))

        assay_step_duration_list.append(step_duration_list)

    df_assay_duration_modified = pd.DataFrame()
    df_assay_duration_modified['Assay'] = assay_name_list
    df_assay_duration_modified['duration'] = assay_step_duration_list

    merged_assay_duration_data = df_assay_duration_modified.merge(df_assay_to_be_scheduled, on="Assay", how="right")

    variable_duration_list = []
    variable_start_step_list = []
    variable_end_step_list = []
    variable_delay_assay_list = []

    # new code ---------------------------------------
    merged_assay_duration_data["compatible_assets"] = assay_list_of_compatible_assets
    merged_assay_duration_data["compatible_assets_binary_variable"] = assay_list_of_compatible_assets_binary_variable
    # end of new code ---------------------------------------

    merged_assay_duration_data[['variable_end_step_4', 'variable_start_step_2']] = [None,None]

    # looping through all the assays - the final merged assay data that consists of the duration too.
    for index in range(0, len(merged_assay_duration_data)):
        assay_duration = merged_assay_duration_data.iloc[index]['duration']
        assay_id = merged_assay_duration_data.iloc[index]['ID']

        step_num = 1
        variable_duration_list_for_assay = []
        variable_start_step_list_for_assay = []
        variable_end_step_list_for_assay = []

        variable_day_start_y_list = []
        variable_shift_start_x_list = []

        # looping through the duration array of each assay
        for duration in assay_duration:
            variable_duration = LpVariable(f'_{assay_id}_duration_{step_num}', 0, None)
            variable_duration_list_for_assay.append(variable_duration)

            day_start_y = LpVariable(f'_{assay_id}_day_start_{index}', cat='Integer')
            variable_day_start_y_list.append(day_start_y)

            shift_start_x = LpVariable(f'_{assay_id}_shift_start_{step_num}', cat='Integer')
            variable_shift_start_x_list.append(shift_start_x)

            if step_num > 1:
                variable_start_step = LpVariable(f'_start_{assay_id}_{step_num}', 0, None)
                variable_start_step_list_for_assay.append(variable_start_step)

            if step_num == 1:
                variable_start_step = forecast_assay_list_step1[index]
                variable_start_step_list_for_assay.append(variable_start_step)

            variable_end_step = LpVariable(f'_end_{assay_id}_{step_num}', 0, None)
            variable_end_step_list_for_assay.append(variable_end_step)

            pdm_scheduler += (variable_end_step == variable_start_step + variable_duration,
                              f'_{assay_id}_{step_num}_end_is_equal_to_start_and_duration')

            step_num += 1

        variable_duration_list.append(variable_duration_list_for_assay)
        variable_start_step_list.append(variable_start_step_list_for_assay)
        variable_end_step_list.append(variable_end_step_list_for_assay)

        # Step 2 should start immediately after Step 1
        pdm_scheduler += (variable_start_step_list_for_assay[1] == variable_end_step_list_for_assay[0],
                          f'_{assay_id}_step2_should_start_immediately_after_step1')

        # Step 3 should start immediately after Step 2
        pdm_scheduler += (variable_start_step_list_for_assay[2] == variable_end_step_list_for_assay[1],
                          f'_{assay_id}_step3_should_start_immediately_after_step2')

        # Step 4 should start after step 3
        pdm_scheduler += (variable_start_step_list_for_assay[3] >= variable_end_step_list_for_assay[2],
                          f'_{assay_id}_step4_should_start_after_step3')

        # Step 5 should start after step 4
        pdm_scheduler += (variable_start_step_list_for_assay[4] >= variable_end_step_list_for_assay[3],
                          f'_{assay_id}_step5_should_start_after_step4')

        # shift start is day start * 3 for all
        pdm_scheduler += (variable_shift_start_x_list[0] == variable_day_start_y_list[0] * 3,
                          f'{assay_id}_shift_start_is_day_start_*_3')

        # step 1 start and step 2 end should happen in the same shift
        pdm_scheduler += (variable_start_step_list_for_assay[0] >= 480 * variable_shift_start_x_list[0],
                          f'{assay_id}_start_time_is_greater_than_shift_start_time')

        # step 2 should be finished in the same shift as step 1
        pdm_scheduler += (variable_end_step_list_for_assay[1] <= 480 * (variable_shift_start_x_list[0] + 1),
                          f'{assay_id}_end_time_step2_less_than_end_of_start_time_of_shift1')

        # step 4 start and step 4 end should happen in the same shift
        pdm_scheduler += (variable_start_step_list_for_assay[3] >= 480 * variable_shift_start_x_list[3],
                          f'{assay_id}_step4_start_time_and_end_time_is_in_same_shift')

        # step 5 should be finished in the same shift as start
        pdm_scheduler += (variable_end_step_list_for_assay[4] <= 480 * (variable_shift_start_x_list[4] + 1),
                          f'{assay_id}_step5_start_time_and_end_time_is_in_same_shift')

        # merged_assay_duration_data.iloc[index]['variable_start_step_2'] = [variable_start_step_list_for_assay[1].__str__()]
        # merged_assay_duration_data.iloc[index]['variable_end_step_4'] = [variable_end_step_list_for_assay[3].__str__()]
        merged_assay_duration_data.loc[index, ['variable_start_step_2']] = variable_start_step_list_for_assay[1]
        merged_assay_duration_data.loc[index, ['variable_end_step_4']] = variable_end_step_list_for_assay[3]

        # # new code ---------------------------------------
        # # code to loop though all the assays
        # # append asset_assay_list to merged_assay_duration_data
        # M = 1000000
        # for i in range(index+1, len(merged_assay_duration_data)):
        #     for asset in merged_assay_duration_data.iloc[index]['compatible_assets']:
        #         if asset in merged_assay_duration_data.iloc[i]['compatible_assets']:
        #             Fijkl = LpVariable(f'_{index}_{i}_{asset}', cat='Binary')
        #             """ _asset__{index}_{asset_name}
        #     """
        #             pdm_scheduler += (____.iloc[index].variable_start_step_list_for_assay[1] >= ____.iloc[i].variable_end_step_list_for_assay[3] - M*(2-merged_assay_duration_data.iloc[i].compatible_assets_binary_variable-merged_assay_duration_data.iloc[index].compatible_assets_binary_variable+Fijkl),
        #                               f'{assay_id}_same_asset_not_involved_in_two_tests')
        #
        # # end of new code ---------------------------------------


        # Delay in tests
        delay_in_assay = LpVariable(f'_{assay_id}_delay', cat='Continuous')

        assay = df_assay_to_be_scheduled[df_assay_to_be_scheduled['ID'] == assay_id]
        assay_due_date = pd.to_datetime(assay['Due Date'].iloc[0]).date()

        business_day_minutes = np.busday_count(today_date, assay_due_date) * 1440

        due_date = LpVariable(f'_{assay_id}_due_date')

        pdm_scheduler += (due_date == business_day_minutes)

        pdm_scheduler += (delay_in_assay == variable_end_step_list_for_assay[4] - due_date,
                          f'_{assay_id}_delay_time')

        variable_delay_assay_list.append(delay_in_assay)

    for index in range(0, len(merged_assay_duration_data)-1):
        # new code ---------------------------------------
        # code to loop though all the assays
        # append asset_assay_list to merged_assay_duration_data
        M = 1000000
        for i in range(index+1, len(merged_assay_duration_data)):
            for asset_index in range(0, len(merged_assay_duration_data.iloc[index]['compatible_assets'])):
                asset = merged_assay_duration_data.iloc[index]['compatible_assets'][asset_index]
                if asset in merged_assay_duration_data.iloc[i]['compatible_assets']:
                    F_index_i_asset = LpVariable(f'_{index}_{i}_{asset}', cat='Binary')
                    """ _asset__{index}_{asset_name}
            """
                    # 1st 2 are an LpVariable
                    pdm_scheduler += (merged_assay_duration_data.iloc[index].variable_start_step_2 >= merged_assay_duration_data.iloc[i].variable_end_step_4
                                      - M*(2 - merged_assay_duration_data.iloc[i].compatible_assets_binary_variable[asset_index] - merged_assay_duration_data.iloc[index].compatible_assets_binary_variable[asset_index] + F_index_i_asset),
                                      f'{index}_{i}_{asset}_same_asset_not_involved_in_two_tests')

        # end of new code ---------------------------------------


    pdm_scheduler += lpSum(variable_delay_assay_list), f"_total_delay" # objective function defined.

    out_dir = "data/output"
    pdm_scheduler.writeLP(f'{out_dir}/pdm-opt-{datetime.now().strftime("%s")}.lp')



"""
for loop - combination of all assays:
    if tests are sharing the same equipment:
        for each shared equipment:
            write the condition
            starts at step 2 -> left side constraint 
            ends step 4 -> right side constraint
            
            variable_end_step_list_for_assay[3]
            
            M (a big number) = 1,000,000

"""
