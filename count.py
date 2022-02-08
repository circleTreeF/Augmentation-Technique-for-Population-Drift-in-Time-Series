import numpy as np
import pandas as pd

raw_data_path = 'raw_data/'


def insert_performance_standard(year, sample=False):
    # TODO: confirm the file name convention
    file_name_prefix = raw_data_path + "standard/annual/historical_data_" + year.__str__() + "/historical_data_" + year.__str__()
    sum = 0
    for quarter in range(1, 5):
        if year == 2021 and quarter >= 2:
            break
        file_name = file_name_prefix + "Q" + quarter.__str__() + ".txt"
        sum = sum + insert_performance_from_txt_with_quarter(file_name, year, quarter, sample)
    return sum


def insert_performance_from_txt_with_quarter(file_name, year, quarter, sample):
    # content_arr = np.genfromtxt(file_name, delimiter='|', dtype=object)
    content_df = pd.read_csv(file_name, delimiter='|', header=None, dtype=object)
    return len(content_df)


sum = 0
for year in range(1999, 2021):
    sum = sum + insert_performance_standard(year)

print(sum)
