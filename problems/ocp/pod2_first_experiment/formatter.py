import pandas as pd
import numpy as np
import datetime

# df['DateTime'] = df.apply(lambda row: datetime.datetime.strptime(row['date']+ ':' + row['time'], '%Y.%m.%d:%H:%M'), axis=1)

# Read csv files into Pandas DataFrames and replace NaN values
def open_csv(filename):
    dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S%z')
    content = pd.read_csv(filename, sep=';',
        parse_dates=['Time'],
        index_col='Time',
        date_parser=dateparse)
    # content = content.interpolate(method='linear', limit_direction='both')
    return content

df = open_csv('power_rack01.csv')

# df = df.resample('M', on='time')['power_p02r01c01srv01'].sum().reset_index()

print(df.index)
print(df)

# df = df.groupby([pd.Grouper(freq='60s', axis=1)])






# df = df.resample('1min')['power_p02r01c01srv01'].ffill()

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(df)









# rs = pd.DataFrame(index=df.resample('1min').iloc[1:].index)

# array of indexes corresponding with closest timestamp after resample
# idx_after = np.searchsorted(df.index.values, rs.index.values)

# print(idx_after)
