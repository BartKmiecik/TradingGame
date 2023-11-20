import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

days = 7
percent_diff = 5
days_to_calculate = 900
crypto = 'EGLD'
data = pd.read_csv(f'{crypto}-USD.csv')
# last_prices = pd.DataFrame(data['Open']).to_numpy()
# last_prices = data.iloc[0].to_numpy()
# print(last_prices)
last_prices = data['1'].to_numpy()
# print(data_to_np)

x = []
for i in range(days):
    x.append(i)

for j in range(days_to_calculate):
    if last_prices[days+1+j] > 0:
        fig, ax = plt.subplots()
        ax.plot(x, last_prices[j:days + j], linewidth=2.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        res = -((last_prices[days+j]-last_prices[days+1+j])/last_prices[days+1+j])*100

        if res > percent_diff:
            plt.savefig(f'Dataset/Rise/{crypto}-USD{days + j}.png', dpi=15)
        elif res < -percent_diff:
            plt.savefig(f'Dataset/Lose/{crypto}-USD{days + j}.png', dpi=15)
        else:
            plt.savefig(f'Dataset/NoChange/{crypto}-USD{days + j}.png', dpi=15)

        plt.close()
        print(f'Day{days+j} last price is: {last_prices[days+j]}, next price: {last_prices[days+j+1]} and percent change: {res}')
        # plt.show()
        # print(res)
        # print(last_prices[days])
