import numpy as np
import pandas as pd


def percent_filter(data: pd.DataFrame = None, array: np.ndarray = None, column: str = None,
                   percent: float = 0.9, step: float = 0.01) -> tuple[float, float] | None:

    if data is not None:
        sample = data[column].to_numpy()
    elif array is not None:
        sample = array
    else:
        raise ValueError("You must provide a DataFrame or a numpy array.")

    sample_mean = np.nanmean(sample)
    sample_max = np.nanmax(sample)
    sample_min = np.nanmin(sample)

    found = False

    upper_bound = sample[sample >= sample_mean]
    lower_bound = sample[sample < sample_mean]

    upper_bound_step = (sample_max - sample_mean) * step
    lower_bound_step = (sample_mean - sample_min) * step

    if upper_bound.shape[0] > lower_bound.shape[0]:
        upper_bound_multiplier = 1
        lower_bound_multiplier = lower_bound.shape[0] / upper_bound.shape[0]
    else:
        upper_bound_multiplier = upper_bound.shape[0] / lower_bound.shape[0]
        lower_bound_multiplier = 1

    lower_cut = sample_mean - lower_bound_step * lower_bound_multiplier
    upper_cut = sample_mean + upper_bound_step * upper_bound_multiplier

    while not found:
        filtered_sample = sample[(sample >= lower_cut) & (sample <= upper_cut)]

        if filtered_sample.shape[0] >= sample.shape[0] * percent:
            found = True
        else:
            lower_cut -= lower_bound_step * lower_bound_multiplier
            upper_cut += upper_bound_step * upper_bound_multiplier

    return lower_cut, upper_cut


def setup_time(data: pd.DataFrame, preset_id: str) -> np.ndarray:
    column_ic, column_fc = 'HoraIC', 'HoraFC'

    filtered_data = data[data["Preset"] == preset_id].copy()

    filtered_data.sort_values(by=column_ic, ascending=True, inplace=True, na_position='last')

    export_data = np.array([])

    for i, IC in enumerate(filtered_data[column_ic]):
        if i == 0:
            continue

        setup = (IC - filtered_data.iloc[i - 1][column_fc]).total_seconds()

        export_data = np.append(export_data, setup)

    return export_data


def n_minus_1_diff(data: pd.DataFrame, column: str) -> np.ndarray:
    data.sort_values(by=column, ascending=True, inplace=True, na_position='last')
    export_data = np.array([])

    for i, current in enumerate(data[column]):
        if i == 0:
            continue

        # print(f"Processing row {i + 1}: {current} - {data.iloc[i - 1]}...")

        diff = (current - data[column].iloc[i - 1]).total_seconds()

        export_data = np.append(export_data, diff)

    return export_data


def compartments(data: pd.DataFrame, mean: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    columns = ('Num Prog', 'HoraEPI', 'HoraSPI')

    progs = data['Num Prog'].unique()

    export_data = list()

    for i, prog in enumerate(progs):

        try:
            print(f"Processing program '{prog}'... ({i + 1}/{len(progs)})")

            prog_data = data[data['Num Prog'] == prog]

            export_data.append({
                'Compartimentos': prog_data[columns[0]].count(),
                'Lead Time': (prog_data[columns[2]].iloc[-1] - prog_data[columns[1]].iloc[-1]).total_seconds(),
                'TipoOperacao': prog_data['TipoOperacao'].iloc[-1],
                # 'HoraEB': prog_data['HoraEB'].iloc[-1],
            })

        except Exception as e:
            print(f"Error processing '{prog}': {e}")
            continue

    else:
        export_data = pd.DataFrame(export_data)

    if mean:
        mean_data_df = export_data.groupby('Compartimentos').mean()
        return export_data, mean_data_df

    else:
        return export_data
