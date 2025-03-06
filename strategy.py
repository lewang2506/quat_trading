import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
import logging
import json
from datetime import datetime, timezone, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PairGridStrategy')

class PairGridStrategy:
    """
    负相关资产对网格交易策略
    
    该策略通过同时双向交易两个负相关品类，根据两者的实时波动幅度动态分配交易
    仓位，在无杠杆条件下设置双向网格，利用价格波动自动触发买卖，通过两个品类
    的价格负相关性实现自对冲，最终在波动中实现风险分散的收益捕获。
    """
    
    def __init__(self, asset_pair: List[str], initial_capital: float, grid_distance_pct: float = 0.05, 
                 max_single_position_pct: float = 0.2, atr_period: int = 14, 
                 hedge_deviation_threshold: float = 0.08, rebalance_period: str = 'W-FRI',
                 circuit_breaker_threshold: float = 0.15):
        """
        初始化策略
        
        Args:
            asset_pair: 交易对，例如 ["BOIL", "KOLD"]
            initial_capital: 初始资金
            grid_distance_pct: 网格间距百分比
            max_single_position_pct: 单个资产最大仓位比例，默认为总资金的20%
            atr_period: ATR计算周期
            hedge_deviation_threshold: 对冲偏离阈值
            rebalance_period: 再平衡周期，默认每周五
            circuit_breaker_threshold: 熔断阈值，默认15%
        """
        self.asset_1 = asset_pair[0]  # BOIL
        self.asset_2 = asset_pair[1]  # KOLD
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.grid_distance_pct = grid_distance_pct
        self.base_grid_distance_pct = grid_distance_pct  # 保存基础网格间距
        self.max_single_position_pct = max_single_position_pct
        self.atr_period = atr_period
        self.hedge_deviation_threshold = hedge_deviation_threshold
        self.rebalance_period = rebalance_period
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # 初始化仓位
        self.positions = {self.asset_1: 0, self.asset_2: 0}
        
        # 初始化波动率权重
        self.volatility_weights = {self.asset_1: 0.5, self.asset_2: 0.5}
        
        # 初始化网格价格
        self.grid_levels = {self.asset_1: [], self.asset_2: []}
        
        # 初始化交易记录
        self.trades = []
        
        # 初始化初始仓位建立标志
        self.initial_position_established = False
        
        # 初始化网格扩展标志
        self.grid_expanded = False
        
        # 初始化当前价格
        self.current_prices = {self.asset_1: None, self.asset_2: None}
        
        # 初始化上一日ATR
        self.previous_day_atr = {self.asset_1: None, self.asset_2: None}
        
        # 当前ATR值
        self.current_atr = {self.asset_1: None, self.asset_2: None}
        
        # 上一次更新ATR的日期
        self.last_atr_update_date = None
        
        # 策略状态
        self.circuit_breaker_triggered = False  # 熔断是否触发
        self.circuit_breaker_asset = None  # 触发熔断的资产
        self.is_active = True  # 策略是否激活
        
        # 价格历史
        self.price_history = {self.asset_1: [], self.asset_2: []}
        
        # 组合价值历史
        self.portfolio_value_history = []
        
        # 预先计算好的ATR值
        self.precalculated_atr = {}
        
        logger.info(f"策略初始化完成，交易对: {asset_pair}, 初始资金: {initial_capital}")
    
    def calculate_atr(self, data: pd.DataFrame, asset: str) -> float:
        """
        计算ATR (Average True Range)
        
        使用最近的atr_period天数据计算滚动平均值
        如果数据不足ATR周期，则使用所有可用数据计算
        
        Args:
            data: 包含OHLC数据的DataFrame
            asset: 资产名称
            
        Returns:
            ATR值
        """
        # 检查是否有预先计算好的ATR值
        current_date = data.index[-1].date()
        if hasattr(self, 'precalculated_atr') and asset in self.precalculated_atr:
            for date_str, atr_value in self.precalculated_atr[asset].items():
                if pd.to_datetime(date_str).date() == current_date:
                    logger.info(f"使用预先计算好的{asset} ATR值: {atr_value:.4f} (日期: {date_str})")
                    return atr_value
        
        # 检查数据是否足够
        if len(data) < 2:
            # 如果数据不足，使用固定值1.20作为ATR
            logger.warning(f"{asset} 数据不足，无法计算ATR，使用固定值: 1.20")
            return 1.20
        
        # 获取最近的数据
        # 如果数据量足够，使用最近的atr_period+1天数据
        # 如果数据量不足，使用所有可用数据
        if len(data) >= self.atr_period + 1:
            recent_data = data.iloc[-(self.atr_period+1):]
        else:
            recent_data = data
            logger.warning(f"{asset} 数据少于ATR周期({len(data)}/{self.atr_period+1})，使用所有可用数据计算")
            
        high = recent_data[f'{asset}_high']
        low = recent_data[f'{asset}_low']
        close = recent_data[f'{asset}_close'].shift(1)
        
        # 处理NaN值
        close = close.fillna(method='bfill')
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # 使用滚动窗口计算ATR
        # 如果数据足够，使用atr_period天的滚动窗口
        # 如果数据不足，使用min_periods=1确保能计算出结果
        atr = tr.rolling(window=self.atr_period, min_periods=1).mean().iloc[-1]
        
        # 记录ATR计算信息，便于调试
        logger.info(f"{asset} ATR计算: 使用{len(recent_data)}天数据, 周期={self.atr_period}, 值={atr:.4f}")
        
        return atr
    
    def update_grid_levels(self, data: pd.DataFrame) -> None:
        """
        更新网格价格水平
        
        Args:
            data: 包含价格数据的DataFrame
        """
        current_date = data.index[-1].date()
        
        # 检查是否需要更新ATR（每日收盘后或极端波动时）
        need_update_atr = self._check_if_need_update_atr(data)
        
        for asset in [self.asset_1, self.asset_2]:
            current_price = data[f'{asset}_close'].iloc[-1]
            self.current_prices[asset] = current_price
            
            if need_update_atr:
                # 计算ATR
                atr = self.calculate_atr(data, asset)
                
                # 保存上一日ATR值（如果是日期变更）
                if self.last_atr_update_date is None or current_date != self.last_atr_update_date:
                    self.previous_day_atr[asset] = self.current_atr.get(asset, atr)
                
                # 更新当前ATR值
                self.current_atr[asset] = atr
                
                # 更新最后ATR更新日期
                self.last_atr_update_date = current_date
                
                # 计算网格间距
                grid_distance = atr * self.grid_distance_pct
                
                # 更新网格价格
                grid_levels = []
                for i in range(-5, 6):  # 创建11个网格层级，从-5到+5
                    grid_price = current_price * (1 + i * grid_distance / current_price)
                    grid_levels.append(grid_price)
                
                # 更新网格价格
                self.grid_levels[asset] = grid_levels
                
                logger.info(f"更新{asset}网格: ATR={atr:.2f}, 网格间距={grid_distance:.2f}, 网格价格={[round(g, 2) for g in grid_levels]}")
            else:
                logger.debug(f"无需更新{asset}的ATR和网格")
    
    def _check_if_need_update_atr(self, data: pd.DataFrame) -> bool:
        """
        检查是否需要更新ATR
        
        条件：
        1. 每日收盘后（强制）
        2. 盘中极端波动（当前TR值超过前一日ATR的2倍）
        
        Args:
            data: 包含价格数据的DataFrame（可以是5分钟数据）
            
        Returns:
            是否需要更新ATR
        """
        current_date = data.index[-1].date()
        current_time = data.index[-1].time()
        
        # 条件1：每日收盘后（强制）
        # 假设收盘时间是16:00，或者是新的一天的第一个数据点
        is_day_end = current_time.hour >= 16 or (self.last_atr_update_date is not None and current_date != self.last_atr_update_date)
        
        # 条件2：盘中极端波动
        is_extreme_volatility = False
        
        # 只有在有上一日ATR值的情况下才检查极端波动
        if all(value is not None for value in self.previous_day_atr.values()):
            for asset in [self.asset_1, self.asset_2]:
                # 使用5分钟数据计算当前的真实波幅(TR)
                if len(data) >= 2:
                    high = data[f'{asset}_high'].iloc[-1]
                    low = data[f'{asset}_low'].iloc[-1]
                    prev_close = data[f'{asset}_close'].iloc[-2]
                    
                    tr1 = high - low
                    tr2 = abs(high - prev_close)
                    tr3 = abs(low - prev_close)
                    
                    current_tr = max(tr1, tr2, tr3)
                    
                    # 检查是否超过前一日ATR的2倍
                    if current_tr > 2 * self.previous_day_atr[asset]:
                        logger.warning(f"{asset} 检测到极端波动: 当前TR={current_tr:.4f}, 前一日ATR={self.previous_day_atr[asset]:.4f}, 比值={current_tr/self.previous_day_atr[asset]:.2f}")
                        is_extreme_volatility = True
                        break
        
        return is_day_end or is_extreme_volatility
    
    def check_rebalance(self, current_date: pd.Timestamp) -> bool:
        """
        检查是否需要再平衡
        
        条件：每周五收盘前1小时（15:00左右）
        
        Args:
            current_date: 当前日期时间
            
        Returns:
            是否需要再平衡
        """
        # 首次运行时需要平衡
        if self.last_atr_update_date is None:
            self.last_atr_update_date = current_date
            return True
        
        # 检查是否是周五且时间接近15:00（收盘前1小时）
        if self.rebalance_period == 'W-FRI':  # 每周五
            is_friday = current_date.dayofweek == 4  # 4 = Friday
            current_time = current_date.time()
            # 检查时间是否在14:55到15:05之间（给定一个小的时间窗口，避免错过）
            is_before_close = (current_time.hour == 14 and current_time.minute >= 55) or \
                             (current_time.hour == 15 and current_time.minute <= 5)
            
            # 获取日期部分进行比较，处理不同类型的日期对象
            current_date_only = current_date.date() if hasattr(current_date, 'date') else current_date
            last_update_date_only = self.last_atr_update_date.date() if hasattr(self.last_atr_update_date, 'date') else self.last_atr_update_date
            
            if is_friday and is_before_close and last_update_date_only != current_date_only:
                logger.info(f"触发周五收盘前再平衡，当前时间: {current_time}")
                self.last_atr_update_date = current_date
                return True
        
        return False
    
    def check_circuit_breaker(self, data: pd.DataFrame) -> bool:
        """
        检查是否触发熔断
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            是否触发熔断
        """
        for asset in [self.asset_1, self.asset_2]:
            if len(data) < 2:
                return False
                
            current_price = data[f'{asset}_close'].iloc[-1]
            previous_price = data[f'{asset}_close'].iloc[-2]
            
            daily_change = abs(current_price - previous_price) / previous_price
            
            if daily_change > self.circuit_breaker_threshold:
                logger.warning(f"{asset}触发熔断: 日内波动{daily_change:.2%} > {self.circuit_breaker_threshold:.2%}")
                self.circuit_breaker_triggered = True
                # 记录触发熔断的资产
                self.circuit_breaker_asset = asset
                # 处理熔断后的逻辑
                self.handle_circuit_breaker(data, asset, current_price, previous_price)
                return True
        
        return False
    
    def handle_circuit_breaker(self, data: pd.DataFrame, triggered_asset: str, current_price: float, previous_price: float) -> None:
        """
        处理熔断后的逻辑
        
        暂停触发熔断的品种当日交易
        
        Args:
            data: 包含价格数据的DataFrame
            triggered_asset: 触发熔断的资产
            current_price: 当前价格
            previous_price: 前一个价格
        """
        # 计算价格变动方向
        price_change = current_price - previous_price
        price_direction = "上涨" if price_change > 0 else "下跌"
        
        # 仅记录熔断信息，不执行反向对冲
        logger.warning(f"触发熔断保护，暂停{triggered_asset}交易，价格{price_direction}幅度: {abs(price_change/previous_price):.2%}")
        
        # 不再执行对冲交易
        # 仅标记熔断状态，在update方法中会跳过该资产的交易
    
    def check_hedge_deviation(self, data: pd.DataFrame) -> bool:
        """
        检查对冲偏离 - 已禁用
        
        此功能已被禁用，不再触发对冲调整
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            始终返回False，不触发对冲调整
        """
        # 此功能已被禁用，直接返回False
        return False
    
    def calculate_portfolio_value(self, data: pd.DataFrame) -> float:
        """
        计算当前组合价值
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            组合总价值
        """
        asset_1_price = data[f'{self.asset_1}_close'].iloc[-1]
        asset_2_price = data[f'{self.asset_2}_close'].iloc[-1]
        
        asset_1_value = self.positions[self.asset_1] * asset_1_price
        asset_2_value = self.positions[self.asset_2] * asset_2_price
        
        # 总价值 = 现金 + 资产价值
        total_value = self.cash + asset_1_value + asset_2_value
        
        return total_value
    
    def execute_trade(self, asset: str, action: str, price: float, quantity: float, reason: str) -> None:
        """
        执行交易并记录
        
        Args:
            asset: 交易资产
            action: 'buy' 或 'sell'
            price: 交易价格
            quantity: 交易数量
            reason: 交易原因
        """
        if quantity <= 0:
            return
            
        trade_value = price * quantity
        
        # 记录交易前的现金余额
        pre_trade_cash = self.cash
        
        # 记录交易前的持仓
        pre_trade_positions = self.positions.copy()
        
        # 计算当前组合总价值
        current_portfolio_value = self.calculate_portfolio_value(pd.DataFrame({
            f'{self.asset_1}_close': [price if asset == self.asset_1 else self.current_prices.get(self.asset_1, price)],
            f'{self.asset_2}_close': [price if asset == self.asset_2 else self.current_prices.get(self.asset_2, price)]
        }))
        
        if action == 'buy':
            # 只有在非初始建仓时才检查资金限制
            if "初始建仓" not in reason:
                if trade_value > self.cash:
                    # 调整数量以匹配可用资金
                    quantity = self.cash / price
                    trade_value = price * quantity
            else:
                # 初始建仓时，按照总资金的20%进行建仓
                target_value = self.initial_capital * self.max_single_position_pct
                quantity = target_value / price
                trade_value = price * quantity
                logger.info(f"{asset} 初始建仓数量设置为 {quantity:.2f}，占总资金的 {self.max_single_position_pct:.0%}")
                
            if quantity <= 0:
                logger.warning(f"{asset} 买入交易取消，数量调整后为0")
                return
                
            self.positions[asset] += quantity
            self.cash -= trade_value
            
        elif action == 'sell':
            if quantity > self.positions[asset]:
                quantity = self.positions[asset]
                trade_value = price * quantity
                
            self.positions[asset] -= quantity
            self.cash += trade_value
        
        # 计算持仓变动
        position_changes = {}
        for asset_name, position in self.positions.items():
            position_changes[asset_name] = position - pre_trade_positions.get(asset_name, 0)
        
        # 确定触发类型
        trigger_type = ""
        if "初始建仓" in reason:
            trigger_type = "初始建仓"
        elif "网格" in reason:
            trigger_type = "网格层触发"
        elif "再平衡" in reason:
            trigger_type = "再平衡触发"
        elif "熔断" in reason:
            trigger_type = "熔断触发"
        elif "对冲" in reason:
            trigger_type = "对冲触发"
        
        # 确定关联网格层
        grid_level = ""
        for part in reason.split():
            if part.startswith(asset + "-L") and ("UP" in part or "DOWN" in part):
                grid_level = part
                break
        
        # 使用回测中的当前时间（最后一个数据点的时间）
        current_timestamp = self.current_timestamp if hasattr(self, 'current_timestamp') else None
        
        # 获取当前ATR值
        atr_value = self.current_atr.get(asset, 0) if hasattr(self, 'current_atr') else 0
        
        # 记录交易
        trade = {
            'timestamp': current_timestamp,
            'symbol': asset,
            'direction': action.upper(),
            'price': price,
            'quantity': int(quantity),
            'trigger_type': trigger_type,
            'grid_level': grid_level,
            'atr_value': atr_value,
            'volatility_weight_ratio': f"{self.volatility_weights[self.asset_1]:.2f}:{self.volatility_weights[self.asset_2]:.2f}",
            'circuit_breaker_active': self.circuit_breaker_triggered,
            'pre_trade_cash': pre_trade_cash,
            'post_trade_cash': self.cash,
            'position_changes': json.dumps(position_changes),
            'reason': reason,
            'boil_price': self.current_prices.get(self.asset_1, 0),
            'kold_price': self.current_prices.get(self.asset_2, 0)
        }
        self.trades.append(trade)
        
        logger.info(f"执行交易: {action} {asset} {quantity:.4f}股 @ {price:.2f}, 价值: {trade_value:.2f}, ATR: {atr_value:.4f}, 原因: {reason}")
    
    def check_grid_signals(self, data: pd.DataFrame) -> None:
        """
        检查网格交易信号
        
        BOIL策略：每下跌一个层距买入，反弹至上一层距卖出
        KOLD策略：每上涨一个层距卖出，回落至下一层距买入
        
        Args:
            data: 包含价格数据的DataFrame
        """
        for asset in [self.asset_1, self.asset_2]:
            # 如果没有网格价格，跳过
            if asset not in self.grid_levels or not self.grid_levels[asset]:
                continue
                
            current_price = data[f'{asset}_close'].iloc[-1]
            previous_price = self.price_history[asset][-1] if self.price_history[asset] else current_price
            
            # 获取网格价格
            grid_prices = sorted(self.grid_levels[asset])
            
            # 找到当前价格所在的网格区间
            current_grid_index = None
            for i in range(len(grid_prices) - 1):
                if grid_prices[i] <= current_price < grid_prices[i + 1]:
                    current_grid_index = i
                    break
            
            # 如果找不到网格区间，跳过
            if current_grid_index is None:
                continue
            
            # 根据资产类型和价格变动方向执行不同的交易策略
            if asset == self.asset_1:  # BOIL策略：下跌买入，上涨卖出
                if current_price < previous_price:  # 价格下跌
                    # 检查是否穿过网格线
                    for i in range(current_grid_index, -1, -1):
                        if previous_price > grid_prices[i] and current_price <= grid_prices[i]:
                            # 价格下穿网格线，买入
                            lower_grid = grid_prices[i-1] if i > 0 else grid_prices[0] * 0.95
                            upper_grid = grid_prices[i]
                            
                            # 计算买入金额（可以根据网格层级调整）
                            buy_value = self.cash * 0.05  # 使用5%的可用资金
                            
                            # 执行买入
                            quantity = buy_value / current_price
                            self.execute_trade(asset, 'buy', current_price, quantity, f"网格下穿: {lower_grid:.2f}-{upper_grid:.2f}")
                
                elif current_price > previous_price:  # 价格上涨
                    # 检查是否穿过网格线
                    for i in range(current_grid_index + 1, len(grid_prices)):
                        if previous_price < grid_prices[i] and current_price >= grid_prices[i]:
                            # 价格上穿网格线，卖出
                            lower_grid = grid_prices[i-1]
                            upper_grid = grid_prices[i]
                            
                            # 计算卖出数量（可以根据网格层级调整）
                            if self.positions[asset] > 0:
                                sell_quantity = self.positions[asset] * 0.2  # 卖出20%的持仓
                                self.execute_trade(asset, 'sell', current_price, sell_quantity, f"网格上穿: {lower_grid:.2f}-{upper_grid:.2f}")
            
            else:  # KOLD策略：上涨卖出，下跌买入
                if current_price > previous_price:  # 价格上涨
                    # 检查是否穿过网格线
                    for i in range(current_grid_index + 1, len(grid_prices)):
                        if previous_price < grid_prices[i] and current_price >= grid_prices[i]:
                            # 价格上穿网格线，卖出
                            lower_grid = grid_prices[i-1]
                            upper_grid = grid_prices[i]
                            
                            # 计算卖出数量（可以根据网格层级调整）
                            if self.positions[asset] > 0:
                                sell_quantity = self.positions[asset] * 0.2  # 卖出20%的持仓
                                self.execute_trade(asset, 'sell', current_price, sell_quantity, f"网格上穿: {lower_grid:.2f}-{upper_grid:.2f}")
                
                elif current_price < previous_price:  # 价格下跌
                    # 检查是否穿过网格线
                    for i in range(current_grid_index, -1, -1):
                        if previous_price > grid_prices[i] and current_price <= grid_prices[i]:
                            # 价格下穿网格线，买入
                            lower_grid = grid_prices[i-1] if i > 0 else grid_prices[0] * 0.95
                            upper_grid = grid_prices[i]
                            
                            # 计算买入金额（可以根据网格层级调整）
                            buy_value = self.cash * 0.05  # 使用5%的可用资金
                            
                            # 执行买入
                            quantity = buy_value / current_price
                            self.execute_trade(asset, 'buy', current_price, quantity, f"网格下穿: {lower_grid:.2f}-{upper_grid:.2f}")
    
    def rebalance_portfolio(self, data: pd.DataFrame) -> None:
        """
        再平衡投资组合
        
        每周五收盘前1小时根据最新波动率重新分配仓位
        不考虑偏离阈值，直接根据计算结果调整
        
        Args:
            data: 包含价格数据的DataFrame
        """
        logger.info("执行投资组合再平衡")
        
        # 重新计算波动率权重
        asset_1_atr = self.calculate_atr(data, self.asset_1)
        asset_2_atr = self.calculate_atr(data, self.asset_2)
        
        total_volatility = asset_1_atr + asset_2_atr
        
        # 计算新的波动率权重
        asset_1_weight = asset_1_atr / total_volatility
        asset_2_weight = asset_2_atr / total_volatility
        
        # 更新波动率权重
        self.volatility_weights[self.asset_1] = asset_1_weight
        self.volatility_weights[self.asset_2] = asset_2_weight
        
        logger.info(f"更新波动率权重: {self.asset_1}={asset_1_weight:.2%}, {self.asset_2}={asset_2_weight:.2%}")
        
        # 计算当前总价值
        total_value = self.calculate_portfolio_value(data)
        
        # 计算目标仓位
        asset_1_target_value = total_value * asset_1_weight
        asset_2_target_value = total_value * asset_2_weight
        
        # 计算当前仓位价值
        asset_1_price = data[f'{self.asset_1}_close'].iloc[-1]
        asset_2_price = data[f'{self.asset_2}_close'].iloc[-1]
        
        asset_1_value = self.positions[self.asset_1] * asset_1_price
        asset_2_value = self.positions[self.asset_2] * asset_2_price
        
        # 调整BOIL仓位 - 不考虑偏离阈值，直接调整
        asset_1_diff = asset_1_target_value - asset_1_value
        if asset_1_diff > 0:
            # 买入BOIL
            buy_quantity = asset_1_diff / asset_1_price
            self.execute_trade(self.asset_1, 'buy', asset_1_price, buy_quantity, "再平衡买入")
        else:
            # 卖出BOIL
            sell_quantity = abs(asset_1_diff) / asset_1_price
            self.execute_trade(self.asset_1, 'sell', asset_1_price, sell_quantity, "再平衡卖出")
        
        # 调整KOLD仓位 - 不考虑偏离阈值，直接调整
        asset_2_diff = asset_2_target_value - asset_2_value
        if asset_2_diff > 0:
            # 买入KOLD
            buy_quantity = asset_2_diff / asset_2_price
            self.execute_trade(self.asset_2, 'buy', asset_2_price, buy_quantity, "再平衡买入")
        else:
            # 卖出KOLD
            sell_quantity = abs(asset_2_diff) / asset_2_price
            self.execute_trade(self.asset_2, 'sell', asset_2_price, sell_quantity, "再平衡卖出")
    
    def hedge_adjustment(self, data: pd.DataFrame) -> None:
        """
        对冲调整 - 已禁用
        
        此功能已被禁用，不再执行对冲调整
        
        Args:
            data: 包含价格数据的DataFrame
        """
        # 此功能已被禁用，不执行任何操作
        logger.info("对冲调整功能已被禁用，不执行任何操作")
        return
    
    def reset_circuit_breaker(self) -> None:
        """重置熔断状态"""
        self.circuit_breaker_triggered = False
        self.circuit_breaker_asset = None
        logger.info("熔断状态已重置")
    
    def establish_initial_position(self, data: pd.DataFrame) -> None:
        """
        执行初始建仓操作，根据波动率权重分配资金
        使用当日开盘价进行交易
        
        Args:
            data: 包含价格数据的DataFrame
        """
        logger.info("执行初始建仓")
        
        # 计算当前总价值
        total_value = self.initial_capital
        
        # 计算波动率权重
        asset_1_atr = self.calculate_atr(data, self.asset_1)
        asset_2_atr = self.calculate_atr(data, self.asset_2)
        
        # 保存当前ATR值
        if not hasattr(self, 'current_atr'):
            self.current_atr = {}
        self.current_atr[self.asset_1] = asset_1_atr
        self.current_atr[self.asset_2] = asset_2_atr
        
        total_volatility = asset_1_atr + asset_2_atr
        
        # 计算波动率权重
        asset_1_weight = asset_1_atr / total_volatility
        asset_2_weight = asset_2_atr / total_volatility
        
        # 更新波动率权重
        self.volatility_weights[self.asset_1] = asset_1_weight
        self.volatility_weights[self.asset_2] = asset_2_weight
        
        logger.info(f"波动率权重计算: {self.asset_1}={asset_1_weight:.2%}, {self.asset_2}={asset_2_weight:.2%}, 总波动率={total_volatility:.4f}")
        
        # 计算目标仓位
        asset_1_target_value = total_value * asset_1_weight
        asset_2_target_value = total_value * asset_2_weight
        
        # 使用开盘价执行交易
        asset_1_price = data[f'{self.asset_1}_open'].iloc[-1]  # 使用开盘价
        asset_2_price = data[f'{self.asset_2}_open'].iloc[-1]  # 使用开盘价
        
        # 保存当前价格
        self.current_prices[self.asset_1] = asset_1_price
        self.current_prices[self.asset_2] = asset_2_price
        
        # 计算交易数量（这里不再计算具体数量，因为在execute_trade中会根据max_single_position_pct计算）
        asset_1_quantity = asset_1_target_value / asset_1_price
        asset_2_quantity = asset_2_target_value / asset_2_price
        
        # 执行交易，标记为初始建仓
        self.execute_trade(self.asset_1, 'buy', asset_1_price, asset_1_quantity, "初始建仓")
        self.execute_trade(self.asset_2, 'buy', asset_2_price, asset_2_quantity, "初始建仓")
        
        # 标记初始建仓已完成
        self.initial_position_established = True
        self.last_atr_update_date = data.index[-1]  # 设置最后再平衡日期为当前日期
        
        logger.info("初始建仓完成")
    
    def update(self, data: pd.DataFrame) -> Dict:
        """
        更新策略状态
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            策略状态摘要
        """
        if not self.is_active:
            return {"status": "inactive"}
            
        current_date = data.index[-1]
        
        # 保存当前时间戳，用于交易记录
        self.current_timestamp = current_date
        
        # 检查熔断
        if self.check_circuit_breaker(data):
            return {
                "status": "circuit_breaker_triggered",
                "triggered_asset": self.circuit_breaker_asset,
                "date": current_date,
                "portfolio_value": self.calculate_portfolio_value(data)
            }
            
        # 更新网格（包含ATR更新逻辑）
        self.update_grid_levels(data)
        
        # 检查是否需要调整网格层距（根据波动率变化）
        self.adjust_grid_levels(data)
        
        # 如果尚未建立初始仓位，执行初始建仓
        if not self.initial_position_established:
            self.establish_initial_position(data)
            return {
                "date": current_date,
                "portfolio_value": self.calculate_portfolio_value(data),
                "status": "initial_position_established"
            }
        
        # 以下操作只在初始建仓完成后执行
        
        # 检查是否需要再平衡
        if self.check_rebalance(current_date):
            self.rebalance_portfolio(data)
        
        # 注意：对冲偏离检查已被禁用
        # 不再执行: if self.check_hedge_deviation(data): self.hedge_adjustment(data)
        
        # 检查网格交易信号（如果熔断已触发，跳过触发熔断的资产的交易）
        if self.circuit_breaker_triggered:
            # 只检查非熔断资产的网格信号
            self.check_grid_signals_for_asset(data, self.asset_2 if self.circuit_breaker_asset == self.asset_1 else self.asset_1)
        else:
            # 正常检查所有资产的网格信号
            self.check_grid_signals(data)
        
        # 更新价格历史
        for asset in [self.asset_1, self.asset_2]:
            self.price_history[asset].append(data[f'{asset}_close'].iloc[-1])
            
        # 更新组合价值历史
        portfolio_value = self.calculate_portfolio_value(data)
        self.portfolio_value_history.append(portfolio_value)
        
        # 返回状态摘要
        return {
            "status": "active",
            "date": current_date,
            "portfolio_value": portfolio_value,
            "positions": self.positions.copy(),
            "cash": self.cash,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "circuit_breaker_asset": self.circuit_breaker_asset
        }
    
    def get_performance_summary(self) -> Dict:
        """
        获取策略表现摘要
        
        Returns:
            策略表现摘要
        """
        if not self.portfolio_value_history:
            return {"status": "no_data"}
            
        initial_value = self.portfolio_value_history[0]
        final_value = self.portfolio_value_history[-1]
        
        total_return = (final_value - initial_value) / initial_value
        
        # 计算日收益率
        daily_returns = []
        for i in range(1, len(self.portfolio_value_history)):
            daily_return = (self.portfolio_value_history[i] - self.portfolio_value_history[i-1]) / self.portfolio_value_history[i-1]
            daily_returns.append(daily_return)
            
        # 计算年化收益率 (假设252个交易日)
        if daily_returns:
            annualized_return = np.mean(daily_returns) * 252
            annualized_volatility = np.std(daily_returns) * np.sqrt(252)
            # 使用2%的无风险利率计算夏普比率
            risk_free_rate = 0.02
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        else:
            annualized_return = 0
            annualized_volatility = 0
            sharpe_ratio = 0
            
        # 计算最大回撤
        max_drawdown = 0
        peak = self.portfolio_value_history[0]
        
        for value in self.portfolio_value_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trade_count": len(self.trades)
        }
    
    def set_precalculated_atr(self, atr_values: Dict[str, float], target_date: str) -> None:
        """
        设置预先计算好的ATR值
        
        Args:
            atr_values: 资产对应的ATR值字典，例如 {'BOIL': 1.5, 'KOLD': 0.8}
            target_date: 目标日期，格式为YYYY-MM-DD
        """
        if not hasattr(self, 'precalculated_atr'):
            self.precalculated_atr = {}
            
        for asset, atr_value in atr_values.items():
            if asset not in self.precalculated_atr:
                self.precalculated_atr[asset] = {}
                
            self.precalculated_atr[asset][target_date] = atr_value
            
        logger.info(f"设置预先计算好的ATR值: {atr_values} (日期: {target_date})")
    
    def set_daily_atr_dict(self, daily_atr_dict: Dict[str, Dict[str, float]]) -> None:
        """
        设置每日ATR值字典
        
        Args:
            daily_atr_dict: 资产对应的每日ATR值字典，例如 
                           {'BOIL': {'2025-01-01': 1.5, '2025-01-02': 1.6}, 
                            'KOLD': {'2025-01-01': 0.8, '2025-01-02': 0.9}}
        """
        if not hasattr(self, 'precalculated_atr'):
            self.precalculated_atr = {}
            
        for asset, atr_values in daily_atr_dict.items():
            if asset not in self.precalculated_atr:
                self.precalculated_atr[asset] = {}
                
            # 更新预先计算的ATR值字典
            self.precalculated_atr[asset].update(atr_values)
        
        # 获取日期范围
        first_asset = next(iter(daily_atr_dict.values()))
        min_date = min(first_asset.keys())
        max_date = max(first_asset.keys())
        
        logger.info(f"已设置每日ATR值字典，资产: {list(daily_atr_dict.keys())}, 日期范围: {min_date} 至 {max_date}")
            
    def adjust_grid_levels(self, data: pd.DataFrame) -> None:
        """
        根据波动率变化调整网格价格间距
        
        当波动率突然增加时，扩大网格间距；当波动率回归正常时，恢复标准网格间距
        使用5分钟数据检测盘中极端波动，无需等待收盘
        
        Args:
            data: 包含价格数据的DataFrame（可以是5分钟数据）
        """
        # 检查是否需要调整网格价格
        # 只有在检测到极端波动时才需要调整
        if self._check_if_need_update_atr(data) and not data.index[-1].time().hour >= 16:
            logger.info("检查是否需要调整网格价格")
            
            # 计算当前ATR（使用5分钟数据）
            asset_1_tr = self._calculate_current_tr(data, self.asset_1)
            asset_2_tr = self._calculate_current_tr(data, self.asset_2)
            
            # 获取当前价格
            asset_1_price = data[f'{self.asset_1}_close'].iloc[-1]
            asset_2_price = data[f'{self.asset_2}_close'].iloc[-1]
            
            # 检查波动率是否显著增加（当前TR与前一日ATR相比）
            asset_1_tr_change = asset_1_tr / self.previous_day_atr.get(self.asset_1, asset_1_tr) - 1
            asset_2_tr_change = asset_2_tr / self.previous_day_atr.get(self.asset_2, asset_2_tr) - 1
            
            # 如果任一资产的TR增加超过100%（即超过前一日ATR的2倍），调整网格间距
            volatility_increased = asset_1_tr_change > 1.0 or asset_2_tr_change > 1.0
            
            # 如果波动率已经增加，但现在回归正常（变化小于-20%），也需要调整
            volatility_normalized = (self.grid_expanded and 
                                   (asset_1_tr_change < -0.2 or asset_2_tr_change < -0.2))
            
            if volatility_increased or volatility_normalized:
                # 调整网格间距
                if volatility_increased and not self.grid_expanded:
                    # 波动率增加，扩大网格间距
                    logger.warning(f"波动率显著增加: {self.asset_1} TR变化={asset_1_tr_change:.2%}, {self.asset_2} TR变化={asset_2_tr_change:.2%}，扩大网格间距")
                    self.grid_distance_pct = self.base_grid_distance_pct * 1.5
                    self.grid_expanded = True
                elif volatility_normalized:
                    # 波动率回归正常，恢复标准网格间距
                    logger.warning(f"波动率回归正常: {self.asset_1} TR变化={asset_1_tr_change:.2%}, {self.asset_2} TR变化={asset_2_tr_change:.2%}，恢复标准网格间距")
                    self.grid_distance_pct = self.base_grid_distance_pct
                    self.grid_expanded = False
                
                # 重新计算网格价格
                self.grid_levels[self.asset_1] = self.calculate_grid_levels(asset_1_price)
                self.grid_levels[self.asset_2] = self.calculate_grid_levels(asset_2_price)
                
                logger.warning(f"调整后的网格价格: {self.asset_1}={[round(g, 2) for g in self.grid_levels[self.asset_1]]}")
                logger.warning(f"调整后的网格价格: {self.asset_2}={[round(g, 2) for g in self.grid_levels[self.asset_2]]}")
    
    def _calculate_current_tr(self, data: pd.DataFrame, asset: str) -> float:
        """
        计算当前的真实波幅(TR)
        
        Args:
            data: 包含价格数据的DataFrame
            asset: 资产名称
            
        Returns:
            当前的真实波幅(TR)
        """
        if len(data) < 2:
            return 0
            
        high = data[f'{asset}_high'].iloc[-1]
        low = data[f'{asset}_low'].iloc[-1]
        prev_close = data[f'{asset}_close'].iloc[-2]
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        return max(tr1, tr2, tr3)
    
    def calculate_grid_levels(self, current_price: float) -> List[float]:
        """
        计算网格价格
        
        Args:
            current_price: 当前价格
            
        Returns:
            网格价格列表
        """
        grid_prices = []
        for i in range(-5, 6):  # 创建11个网格层级，从-5到+5
            grid_price = current_price * (1 + i * self.grid_distance_pct / current_price)
            grid_prices.append(grid_price)
        return grid_prices
    
    def check_grid_signals_for_asset(self, data: pd.DataFrame, asset: str) -> None:
        """
        检查单个资产的网格交易信号
        
        Args:
            data: 包含价格数据的DataFrame
            asset: 要检查的资产
        """
        # 如果没有网格价格，跳过
        if asset not in self.grid_levels or not self.grid_levels[asset]:
            return
            
        current_price = data[f'{asset}_close'].iloc[-1]
        previous_price = self.price_history[asset][-1] if self.price_history[asset] else current_price
        
        # 获取网格价格
        grid_prices = sorted(self.grid_levels[asset])
        
        # 找到当前价格所在的网格区间
        current_grid_index = None
        for i in range(len(grid_prices) - 1):
            if grid_prices[i] <= current_price < grid_prices[i + 1]:
                current_grid_index = i
                break
        
        # 如果找不到网格区间，跳过
        if current_grid_index is None:
            return
        
        # 根据资产类型和价格变动方向执行不同的交易策略
        if asset == self.asset_1:  # BOIL策略：下跌买入，上涨卖出
            if current_price < previous_price:  # 价格下跌
                # 检查是否穿过网格线
                for i in range(current_grid_index, -1, -1):
                    if previous_price > grid_prices[i] and current_price <= grid_prices[i]:
                        # 价格下穿网格线，买入
                        lower_grid = grid_prices[i-1] if i > 0 else grid_prices[0] * 0.95
                        upper_grid = grid_prices[i]
                        
                        # 新逻辑：买入数量为当前持仓的50%
                        if self.positions[asset] > 0:
                            buy_quantity = self.positions[asset] * 0.5  # 买入50%的持仓
                            self.execute_trade(asset, 'buy', current_price, buy_quantity, f"网格下穿: {lower_grid:.2f}-{upper_grid:.2f}")
                        else:
                            # 如果没有持仓，使用5%的可用资金
                            buy_value = self.cash * 0.05
                            quantity = buy_value / current_price
                            self.execute_trade(asset, 'buy', current_price, quantity, f"网格下穿: {lower_grid:.2f}-{upper_grid:.2f}")
        
        elif current_price > previous_price:  # 价格上涨
            # 检查是否穿过网格线
            for i in range(current_grid_index + 1, len(grid_prices)):
                if previous_price < grid_prices[i] and current_price >= grid_prices[i]:
                    # 价格上穿网格线，卖出
                    lower_grid = grid_prices[i-1]
                    upper_grid = grid_prices[i]
                    
                    # 新逻辑：卖出数量为当前持仓的50%
                    if self.positions[asset] > 0:
                        sell_quantity = self.positions[asset] * 0.5  # 卖出50%的持仓
                        self.execute_trade(asset, 'sell', current_price, sell_quantity, f"网格上穿: {lower_grid:.2f}-{upper_grid:.2f}")
        
        else:  # KOLD策略：上涨卖出，下跌买入
            if current_price > previous_price:  # 价格上涨
                # 检查是否穿过网格线
                for i in range(current_grid_index + 1, len(grid_prices)):
                    if previous_price < grid_prices[i] and current_price >= grid_prices[i]:
                        # 价格上穿网格线，卖出
                        lower_grid = grid_prices[i-1]
                        upper_grid = grid_prices[i]
                        
                        # 新逻辑：卖出数量为当前持仓的50%
                        if self.positions[asset] > 0:
                            sell_quantity = self.positions[asset] * 0.5  # 卖出50%的持仓
                            self.execute_trade(asset, 'sell', current_price, sell_quantity, f"网格上穿: {lower_grid:.2f}-{upper_grid:.2f}")
            
            elif current_price < previous_price:  # 价格下跌
                # 检查是否穿过网格线
                for i in range(current_grid_index, -1, -1):
                    if previous_price > grid_prices[i] and current_price <= grid_prices[i]:
                        # 价格下穿网格线，买入
                        lower_grid = grid_prices[i-1] if i > 0 else grid_prices[0] * 0.95
                        upper_grid = grid_prices[i]
                        
                        # 新逻辑：买入数量为当前持仓的50%
                        if self.positions[asset] > 0:
                            buy_quantity = self.positions[asset] * 0.5  # 买入50%的持仓
                            self.execute_trade(asset, 'buy', current_price, buy_quantity, f"网格下穿: {lower_grid:.2f}-{upper_grid:.2f}")
                        else:
                            # 如果没有持仓，使用5%的可用资金
                            buy_value = self.cash * 0.05
                            quantity = buy_value / current_price
                            self.execute_trade(asset, 'buy', current_price, quantity, f"网格下穿: {lower_grid:.2f}-{upper_grid:.2f}") 