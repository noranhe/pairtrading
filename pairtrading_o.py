from typing import List, Dict
from vnpy.app.portfolio_strategy import StrategyTemplate, StrategyEngine
from vnpy.trader.utility import BarGenerator, ArrayManager
from vnpy.trader.object import TickData, BarData
from vnpy.trader.constant import Interval
from datetime import datetime 
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from patsy import dmatrices


class PairTradingStrategy(StrategyTemplate):
    """"""

    author = ""

    bar_interval = 30
    bar_frequency = Interval.MINUTE

    fixed_size = 1
    price_add = 5
    boll_dev = 1.5

    df = []
    mean = []
    std = []

    boll_up = 0
    boll_down = 0
    boll_mean = 0
    boll_std = 0
    spread_mean = 0
    spread_std = 0
    spread_value = 0
    y_pos = 0
    x_pos = 0
    y_target = 0
    x_target = 0


    parameters = [
        "boll_dev",
        "fixed_size",
        "price_add"
    ]

    variables = [
        "boll_up",
        "boll_down",
        "boll_mean",
        "boll_std",
        "spread_mean",
        "spread_std",
        "spread_value",
        "y_pos",
        "x_pos",
        "y_target",
        "x_target"
    ]

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: List[str],
        setting: dict
    ):
        """"""
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)

        self.bgs: Dict[str, BarGenerator] = {}
        self.last_tick_time: datetime = None

        for vt_symbol in self.vt_symbols: 
    
            def on_bar(bar: BarData):
                """"""
                pass

            self.bgs[vt_symbol] = BarGenerator(on_bar)

        # 单品种K线合成class
        self.bars = {}
        for vt_symbol in self.vt_symbols:
            self.bars[vt_symbol] = Single_bar(self,vt_symbol)

        self.y_symbol = vt_symbols[0]
        self.x_symbol = vt_symbols[1]  
        
        # 单品种K线获取接口
        self.am1 = self.bars[self.y_symbol].am
        self.am2 = self.bars[self.x_symbol].am
        
        # 用来计算是否合成了一根bar
    
        self.bar_count = 1 
    
        self.df = []
        self.mean = []
        self.std = []
    
    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

        self.load_bars(242)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        if (
            self.last_tick_time
            and self.last_tick_time.minute != tick.datetime.minute
        ):
            bars = {}
            for vt_symbol, bg in self.bgs.items():
                bars[vt_symbol] = bg.generate()
            self.on_bars(bars)

        bg: BarGenerator = self.bgs[tick.vt_symbol]
        bg.update_tick(tick)

        self.last_tick_time = tick.datetime

   
    def on_bars(self, bars: Dict[str, BarData]):
        """"""
        # 更新1分钟k线
        for vt_symbol in bars:
            
            self.bars[vt_symbol].on_bar(bars[vt_symbol])

        y_bar_count = self.bars[self.y_symbol].bar_count
        x_bar_count = self.bars[self.x_symbol].bar_count
        
        #　判断逻辑
        if self.bar_count == y_bar_count and self.bar_count == x_bar_count:
            self.bar_count += 1
            self.on_30min_bars(bars)
            
    def on_30min_bars(self, bars: Dict[str, BarData]):
        """"""
        self.cancel_all()
       
        am1 = self.am1
        am2 = self.am2

        bar1 = bars[self.y_symbol]
        bar2 = bars[self.x_symbol]

        if not am1.inited or not am2.inited:
            return
        
        self.df = pd.concat([pd.Series(am1.close),pd.Series(am2.close)], axis=1).dropna()
        self.df.columns = [self.y_symbol.split('.')[0],self.x_symbol.split('.')[0]]

        y, X = patsy.dmatrices('%s ~ %s+0'%(self.df.columns[0],self.df.columns[1]), data = self.df)

        mod = sm.OLS(y,X)
        result=mod.fit()

        self.spread_array = result.resid
        self.spread_value = self.spread_array[-1]

        self.spread_mean = self.spread_array.mean()
        self.mean.append(self.spread_mean)
        self.boll_mean = self.mean[-1]

        self.spread_std = self.spread_array.std()
        self.std.append(self.spread_std)
        self.boll_std = self.std[-1]

        self.boll_up = self.boll_mean + self.boll_std * self.boll_dev
        self.boll_down = self.boll_mean - self.boll_std * self.boll_dev    

        # 查询当前持仓
        self.y_pos = self.get_pos(self.y_symbol)
        self.x_pos = self.get_pos(self.x_symbol)

        # 计算目标仓位
        if self.y_pos ==0:
            if self.spread_value > self.boll_up:
                self.y_target = 8*self.fixed_size
                self.x_target = -10*self.fixed_size
            elif self.spread_value < self.boll_down:
                self.y_target = -8*self.fixed_size
                self.x_target = 10*self.fixed_size
        elif self.y_pos > 0:
            if self.spread_value <= self.boll_mean:
                self.y_target = 0
                self.x_target = 0         
        else:
            if self.spread_value >= self.boll_mean:
                self.y_target = 0
                self.x_target = 0

        target = {
            self.y_symbol: self.y_target,
            self.x_symbol: self.x_target,
        }

        # 执行交易委托
        for vt_symbol in self.vt_symbols:
            target_pos = target[vt_symbol]
            current_pos = self.get_pos(vt_symbol)

            pos_diff = target_pos - current_pos
            volume = abs(pos_diff)
            bar = bars[vt_symbol]

            if pos_diff > 0:
                price = bar.close_price + self.price_add
                
                if current_pos < 0:
                    self.cover(vt_symbol, price, volume)
                else:
                    self.buy(vt_symbol, price, volume)
            elif pos_diff < 0:
                price = bar.close_price - self.price_add

                if current_pos > 0:
                    self.sell(vt_symbol, price, volume)
                else:
                    self.short(vt_symbol, price, volume)

        self.put_event()

    
class Single_bar:
    
    """
    用来生成单品种的K线
    
    """

    def __init__(self, strategy:StrategyTemplate, vt_symbol:str):
        """"""
        # 和portfolio的接口
        self.portfolio= strategy
        self.vt_symbol = vt_symbol
        
        # 需要合成的K线周期设置
        self.bar_interval = self.portfolio.bar_interval
        self.bar_frequency = self.portfolio.bar_frequency

        # K线合成工具和储存容器
        self.bg = BarGenerator(self.on_bar, self.bar_interval, self.on_30min_bar,self.bar_frequency)
        self.am = ArrayManager()
        
        # K线合成计数
        self.bar_count  = 0

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.bg.update_bar(bar)

    def on_30min_bar(self, bar: BarData):
        """"""
        self.bar_count += 1
        self.am.update_bar(bar)