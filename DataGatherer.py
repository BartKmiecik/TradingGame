import requests
import pandas as pd
import numpy as np
api = '5bd396898ffb49d768557dffd57a56df18653f0834be43e6a1939d0ebf1488fb'
currency1, currnecy2 = 'EGLD', 'USD'
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = F'https://min-api.cryptocompare.com/data/v2/histoday?fsym={currency1}&tsym={currnecy2}&limit=1000&apikey={api}'
r = requests.get(url)
data = r.json()
data_list = data['Data']['Data']
# print(data['Data']['Data'])
to_save = []
for i in range(len(data_list)):
    to_save.append([i, data_list[i]['open']])
    print(data_list[i]['open'])

to_save = pd.DataFrame(to_save)

# data_frame = pd.DataFrame(data["Time Series (Digital Currency Daily)"])
# data_frame = data_frame.loc[:, ::-1]
# numpy_array = data_frame.iloc[0].to_numpy()

# print(numpy_array[0])
to_save2 = to_save.to_csv(f'{currency1}-{currnecy2}.csv', index=False)