import os
from utils import *


DATA_PATH = 'data/'
EXPORT_PATH = 'export/'
FILE_NAMES = [f for f in os.listdir(DATA_PATH) if f.endswith('.xlsx')]


if __name__ == '__main__':

    for FILE_NAME in FILE_NAMES:

        print(f"Processing file '{FILE_NAME}'...")

        print(f"Reading data...", end=' ')

        df = pd.read_excel(DATA_PATH + FILE_NAME).dropna(subset=['Centro'])

        print(f"OK\nFiltering products...", end=' ')

        default_conditions = (df['Modo'] == 'Auto')

        df = df[default_conditions]

        load_conditions = ((df['TipoOperacao'] == 'Carga TOP') | (df['TipoOperacao'] == 'Carga Botton'))

        unload_conditions = (df['TipoOperacao'] == 'Descarga')

        load_df = df[load_conditions]
        unload_df = df[unload_conditions]

        # PERCENT FILTER
        """

        print(f"OK\nObtaining bounders values...", end=' ')

        columns = ['Quantidade (l)', 'Vazao (l/min)']

        for i, data in enumerate((load_df, unload_df)):
            if i == 0:
                products = load_df['Nome Produto'].unique()
                data_type = 'Carreg'
            else:
                products = unload_df['Nome Produto'].unique()
                data_type = 'Descarreg'

            print(f'Processing {data_type} data:')

            for column in columns:
                print(f"\tFiltering by '{column}':")
                if column == 'Quantidade (l)':
                    suffix = 'Volume'
                else:
                    suffix = 'Vazao'

                # quantidade_min, quantidade_max = percent_filter(
                #     data=data, column=column, percent=0.9
                # )
                # 
                # filtro = ((data[column] >= quantidade_min) & (data[column] <= quantidade_max))
                # export_data = data[filtro][column]
                # 
                # export_data.to_csv(
                #     f'{EXPORT_PATH}{FILE_NAME.split('.')[0]}_{data_type}_{suffix}.txt',
                #     index=False,
                #     header=False,
                #     sep='\t'
                # )
                # 
                # print(f"OK")

                for product in products:
                    print(f"\t\tProduct: {product}...", end=' ')

                    product_data = data[data['Nome Produto'] == product]

                    quantidade_min, quantidade_max = percent_filter(
                        data=product_data, column=column, percent=0.9
                    )

                    filtro = ((product_data[column] >= quantidade_min) & (product_data[column] <= quantidade_max))
                    export_data = product_data[filtro][column]

                    export_data.to_csv(
                        f'{EXPORT_PATH}{FILE_NAME.split('.')[0]}_{product}_{data_type}_{suffix}.txt',
                        index=False,
                        header=False,
                        sep='\t'
                    )

                    print(f"OK")

        """

        # SETUP TIME
        """
        
        print(f"OK\nFiltering data...", end=' ')
    
        load_presets = sorted(load_df['Preset'].unique())
        unload_presets = sorted(unload_df['Preset'].unique())
        load_setup_times = np.array([])
        unload_setup_times = np.array([])
    
        for preset in load_presets:
            setup_array = setup_time(load_df, preset)
            setup_min, setup_max = percent_filter(array=setup_array, percent=0.9)
            load_setup_times = np.concatenate(
                (load_setup_times, setup_array[(setup_array >= setup_min) & (setup_array <= setup_max)])
            )
    
        for preset in unload_presets:
            setup_array = setup_time(unload_df, preset)
            setup_min, setup_max = percent_filter(array=setup_array, percent=0.9)
            unload_setup_times = np.concatenate(
                (unload_setup_times, setup_array[(setup_array >= setup_min) & (setup_array <= setup_max)])
            )
    
        print(f"OK\nExporting data...", end=' ')
    
        np.savetxt(EXPORT_PATH + 'BABET_' + 'load_setup_times.txt', load_setup_times, fmt='%i', delimiter='\n')
        np.savetxt(EXPORT_PATH + 'BABET_' + 'unload_setup_times.txt', unload_setup_times, fmt='%i', delimiter='\n')
    
        print(f"OK\nData exported to '{EXPORT_PATH}' path.")
        
        """

        # COMPARTMENTS
        # """

        print(f"OK\nFiltering data...\n", end=' ')

        lead_time_df = compartments(df, mean=False) #, mean_lead_time_df
        # eb_df = compartments(df)

        print(f"OK\nExporting data...", end=' ')

        lead_time_df.to_excel(EXPORT_PATH + FILE_NAME.split('.')[0] + '_lead_time_TO.xlsx', index=False)
        # mean_lead_time_df.to_excel(EXPORT_PATH + FILE_NAME.split('.')[0] + '_mean_lead_time_TO.xlsx', index=True)

        print(f"OK\nData exported to '{EXPORT_PATH}' path.")

        eb_results = n_minus_1_diff(eb_df, 'HoraEB')
        # np.savetxt(EXPORT_PATH + FILE_NAME.split('.')[0] + '_truck_inbound.txt', eb_results, fmt='%i', delimiter='\n')

        # """

    print("Process finished.")
