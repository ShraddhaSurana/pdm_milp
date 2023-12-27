import pandas as pd
from pulp import LpVariable, lpSum, LpProblem, LpMinimize

pdm_scheduler = LpProblem('work_package_assay_scheduling', LpMinimize)


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def create_assay_constraints(df_assay_to_be_scheduled, df_asset_assay_mapping, pdm_scheduler=None):
    asset_test_mapping = df_asset_assay_mapping.groupby("assay")["assetid"].agg(list).reset_index()
    asset_test_mapping.rename(columns={'assay': 'Assay'}, inplace=True)

    merged_assay_data = pd.merge(
        asset_test_mapping,
        df_assay_to_be_scheduled,
        on="Assay",
        how="right",
    )

    merged_assay_data['assay_asset_id'] = merged_assay_data.apply(
        lambda row: [f"asset_{row['ID']}_{asset}" for asset in row['assetid']],
        axis=1
    )

    for index, assay_asset_id_list in enumerate(merged_assay_data['assay_asset_id']):
        assay_asset_binary_variable_list = [
            LpVariable(f'_{assay_asset_id}', cat='Binary') for assay_asset_id in assay_asset_id_list
        ]

        pdm_scheduler += lpSum(assay_asset_binary_variable_list) == 1, f"_{index}_Only one asset can be assigned"


if __name__ == "__main__":
    df_asset_assay_mapping = load_data("data/asset_assay_mapping.csv")
    df_assay_to_be_scheduled = load_data("data/assay_to_be_scheduled.csv")

    create_assay_constraints(df_assay_to_be_scheduled, df_asset_assay_mapping, pdm_scheduler)
