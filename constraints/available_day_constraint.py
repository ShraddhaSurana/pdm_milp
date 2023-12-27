from datetime import datetime

import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def create_variables(df_assay_to_be_scheduled):
    today_date = datetime.today().date()
    forecast_assay_list_step1 = []
    available_dates = []
    business_days_list = []
    unique_id = []

    pdm_scheduler = LpProblem('work_package_assay_scheduling', LpMinimize)

    for row in range(len(df_assay_to_be_scheduled)):
        unique_id.append(str(row))

        start = f"_start_{row}_1"
        var_start = LpVariable(start, 0, None)
        forecast_assay_list_step1.append(var_start)

        available_date = pd.to_datetime(df_assay_to_be_scheduled.iloc[row]['Available date']).date()
        available_dates.append(available_date)

        business_days = max(np.busday_count(today_date, available_date) * 1440, 0)
        business_days_list.append(business_days)

    df_assay_to_be_scheduled['ID'] = unique_id

    return pdm_scheduler, forecast_assay_list_step1, available_dates, business_days_list


def add_constraints(pdm_scheduler, forecast_assay_list_step1, business_days_list):
    for index, var_start in enumerate(forecast_assay_list_step1):
        pdm_scheduler += var_start >= business_days_list[index], f'_{var_start}_cannot_start_before_avail'


if __name__ == "__main__":
    df_assay_to_be_scheduled = load_data("../data/assay_to_be_scheduled.csv")

    pdm_scheduler, forecast_assay_list_step1, available_dates, business_days_list = create_variables(
        df_assay_to_be_scheduled)
    add_constraints(pdm_scheduler, forecast_assay_list_step1, business_days_list)

    print(forecast_assay_list_step1)
    print(available_dates)
    print(business_days_list)
