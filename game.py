import pandas as pd
import copy
class TradeGame():
    def __init__(self, data : pd.DataFrame, money : float, transaction_fee = .3, trade_time = -1, custom_close_name = 'Close'):
        self.data = data
        self.starting_money = money
        self.current_money = self.starting_money
        self.trade_times = trade_time if trade_time > 0 else data.__len__() - 1
        self.transaction_fee = 1 - transaction_fee / 100 # transaction fee, default 0.3%
        self.day = 0
        self.long_buy = []
        self.short_buy = []
        self.close = custom_close_name
        self.current_price = self.data[self.close][self.day]

    def next_day(self):
        self.day += 1
        self.current_price = self.data[self.close][self.day]

    def reset(self):
        self.day += 1
        self.current_price = self.data[self.close][self.day]
        self.current_money = self.starting_money

    def Buy(self, amount_to_spend, direction):
            price = self.data[self.close][self.day]
            amount_to_spend = self.current_money if amount_to_spend > self.current_money else amount_to_spend
            if self.current_money > 0:
                if(direction <= 0):
                    for bp, sa in self.long_buy:
                        self.current_money += sa * price * self.transaction_fee
                    self.long_buy.clear()
                    amount_to_spend = amount_to_spend if amount_to_spend > 0 else self.current_money
                    stocks_short = (amount_to_spend * self.transaction_fee) / price
                    self.short_buy.append((price, stocks_short))
                    self.current_money -= amount_to_spend
                else:
                    for bp, sa in self.short_buy:
                        p1 = bp * sa
                        p2 = price * sa
                        self.current_money += (p1 + (p1 - p2)) * self.transaction_fee
                    self.short_buy.clear()
                    amount_to_spend = amount_to_spend if amount_to_spend > 0 else self.current_money
                    stocks_long = (amount_to_spend * self.transaction_fee) / price
                    self.long_buy.append((price, stocks_long))
                    self.current_money -= amount_to_spend

    def Stand(self):
        pass

    def CalculateWholeResult(self):
        price = self.data[self.close][self.day]
        money = 0
        for bp, sa in self.long_buy:
            money += sa * price * self.transaction_fee
        for bp, sa in self.short_buy:
            p1 = bp * sa
            p2 = price * sa
            money += (p1 + (p1 - p2)) * self.transaction_fee
        money += self.current_money
        return money



    """def calculate_reward(self, action):
        t1 = self.data[self.close][self.day]
        t2 = self.data[self.close][self.day + 1]
        diff = t2 - t1
        if diff >= 0:
            if diff >= 3:
                step = 2
                if action == step:
                    return 1.0
            else:
                step = 1
                if action == step:
                    return 0.0
        else:
            if diff <= -3:
                step = 0
                if action == step:
                    return 1.0
            else:
                step = 1
                if action == step:
                    return 0.0
        return -1.0"""


    def play_step(self, action):
        match int(action):
            case 1:
                while True:
                    try:
                        amount_to_spend = float(input("\nHow much you like to invest? (x =< 0 - all money)\n"))
                        break
                    except:
                        print("That's not a valid option!\nTry again: (x =< 0 - all money)")
                while True:
                    try:
                        direction = int(input('\nLong (x > 0) or short (x <= 0)?\n'))
                        break
                    except:
                        print("That's not a valid option!\nTry again: Long (x > 0) or short (x <= 0)")
                self.Buy(amount_to_spend, direction)
            case 0:
                self.Stand()
        self.next_day()
        return

    def play_game(self):
        while self.day < self.trade_times:
            print(f'Day: {self.day} || Stock price: {self.current_price} || Portfolio value: {self.CalculateWholeResult()}')
            while True:
                try:
                    action = int(input('\n1 - Buy stocks || 0 - Stand\n'))
                    break
                except:
                    print("That's not a valid option!\nTry again: 1 - Buy stocks || 0 - Stand")

            self.play_step(action)

print(f'\n\n\n\nWelcome\n----------------------------\nThis game will simulate trading stock, crypto, etc....\n'
      f'There 2 possible actions: 0 - Stand | 1 - Buy (after that you input how much you want to spend, you\'l choose direction\n'
      f'if you have stocks long stocks and choose to buy shrot all long stocks will be sold and viceversa\n')

data = pd.read_csv('BTC-USD.csv')
money = 1000

game = TradeGame(data,money)
game.play_game()

