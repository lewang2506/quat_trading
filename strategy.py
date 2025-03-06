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
    
    def __init__(
        self,
        asset_pair: Tuple[str, str],
        initial_capital: float,
        atr_period: int = 14,
        grid_distance_pct: float = 0.5,  # ATR的百分比
        max_single_position_pct: float = 0.2,  # 最大单边仓位占总资金的百分比
        hedge_deviation_threshold: float = 0.05,  # 对冲偏离阈值
        rebalance_period: str = 'W-FRI',  # 每周五再平衡
        circuit_breaker_threshold: float = 0.15,  # 熔断条件：单日波幅>15%
    ):
        """
        初始化策略参数
        
        Args:
            asset_pair: 交易的资产对，如 ('BOIL', 'KOLD')
            initial_capital: 初始资金
            atr_period: 计算ATR的周期
            grid_distance_pct: 网格间距为ATR的百分比
            max_single_position_pct: 最大单边仓位占总资金的百分比
            hedge_deviation_threshold: 对冲偏离阈值
            rebalance_period: 再平衡周期，默认每周五
            circuit_breaker_threshold: 熔断阈值
        """
        self.asset_1, self.asset_2 = asset_pair
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.atr_period = atr_period
        self.grid_distance_pct = grid_distance_pct
        self.max_single_position_pct = max_single_position_pct
        self.hedge_deviation_threshold = hedge_deviation_threshold
        self.rebalance_period = rebalance_period
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # 初始建仓标志
        self.initial_position_established = False
        
        # 持仓信息
        self.positions = {
            self.asset_1: 0,  # 持仓数量
            self.asset_2: 0,  # 持仓数量
        }
        
        # 网格信息
        self.grids = {
            self.asset_1: [],  # 价格网格
            self.asset_2: [],  # 价格网格
        }
        
        # 交易记录
        self.trades = []
        
        # 资产价格历史
        self.price_history = {
            self.asset_1: [],
            self.asset_2: [],
        }
        
        # 资产价值历史
        self.portfolio_value_history = []
        
        # 当前价格
        self.current_prices = {
            self.asset_1: None,
            self.asset_2: None,
        }
        
        # 上次再平衡日期
        self.last_rebalance_date = None
        
        # 策略状态
        self.is_active = True  # 策略是否激活
        self.circuit_breaker_triggered = False  # 熔断是否触发
        
        # 波动率权重
        self.volatility_weights = {self.asset_1: 0.5, self.asset_2: 0.5}
        
        # 当前ATR值
        self.current_atr = {self.asset_1: 0, self.asset_2: 0}
        
        logger.info(f"策略初始化完成，交易对: {asset_pair}, 初始资金: {initial_capital}")
    
    def calculate_atr(self, data: pd.DataFrame, asset: str) -> float:
        """
        计算ATR (Average True Range)
        
        Args:
            data: 包含OHLC数据的DataFrame
            asset: 资产名称
            
        Returns:
            ATR值
        """
        # 检查数据是否足够
        if len(data) < 2:
            # 如果数据不足，使用固定值1.20作为ATR
            logger.warning(f"{asset} 数据不足，无法计算ATR，使用固定值: 1.20")
            return 1.20
            
        high = data[f'{asset}_high']
        low = data[f'{asset}_low']
        close = data[f'{asset}_close'].shift(1)
        
        # 处理NaN值
        close = close.fillna(method='bfill')
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # 如果数据少于ATR周期，使用所有可用数据
        available_periods = min(len(tr), self.atr_period)
        if available_periods < self.atr_period:
            logger.warning(f"{asset} 数据少于ATR周期({available_periods}/{self.atr_period})，使用可用数据计算")
            
        atr = tr.rolling(window=available_periods, min_periods=1).mean().iloc[-1]
        
        # 记录ATR计算信息，便于调试
        logger.info(f"{asset} ATR计算: 周期={available_periods}, 值={atr:.4f}, 数据长度={len(data)}")
        
        return atr
    
    def update_grid_levels(self, data: pd.DataFrame) -> None:
        """
        更新网格价格水平
        
        Args:
            data: 包含价格数据的DataFrame
        """
        for asset in [self.asset_1, self.asset_2]:
            current_price = data[f'{asset}_close'].iloc[-1]
            self.current_prices[asset] = current_price
            
            # 计算ATR
            atr = self.calculate_atr(data, asset)
            # 更新当前ATR值
            self.current_atr[asset] = atr
            grid_distance = atr * self.grid_distance_pct
            
            # 创建网格 (上下各5层)
            grid_levels = []
            for i in range(-5, 6):
                level = current_price * (1 + i * grid_distance / current_price)
                grid_levels.append(level)
            
            self.grids[asset] = sorted(grid_levels)
            
            logger.info(f"更新{asset}网格: ATR={atr:.2f}, 网格间距={grid_distance:.2f}, 网格价格={[round(g, 2) for g in grid_levels]}")
    
    def check_rebalance(self, current_date: pd.Timestamp) -> bool:
        """
        检查是否需要再平衡
        
        Args:
            current_date: 当前日期
            
        Returns:
            是否需要再平衡
        """
        # 首次运行时需要平衡
        if self.last_rebalance_date is None:
            self.last_rebalance_date = current_date
            return True
        
        # 按照再平衡周期检查
        if self.rebalance_period == 'W-FRI':  # 每周五
            if current_date.dayofweek == 4 and self.last_rebalance_date.dayofweek != 4:  # 4 = Friday
                self.last_rebalance_date = current_date
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
                return True
        
        return False
    
    def check_hedge_deviation(self, data: pd.DataFrame) -> bool:
        """
        检查对冲偏离
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            是否需要对冲调整
        """
        if len(self.portfolio_value_history) == 0:
            return False
            
        initial_value = self.portfolio_value_history[0]
        current_value = self.calculate_portfolio_value(data)
        
        deviation = abs(current_value - initial_value) / initial_value
        
        if deviation > self.hedge_deviation_threshold:
            logger.warning(f"触发对冲调整: 组合价值偏离{deviation:.2%} > {self.hedge_deviation_threshold:.2%}")
            return True
            
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
        
        cash = self.current_capital - asset_1_value - asset_2_value
        total_value = cash + asset_1_value + asset_2_value
        
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
        pre_trade_cash = self.current_capital
        
        # 记录交易前的持仓
        pre_trade_positions = self.positions.copy()
        
        if action == 'buy':
            if trade_value > self.current_capital:
                # 调整数量以匹配可用资金
                quantity = self.current_capital / price
                trade_value = price * quantity
                
            self.positions[asset] += quantity
            self.current_capital -= trade_value
        elif action == 'sell':
            if quantity > self.positions[asset]:
                quantity = self.positions[asset]
                trade_value = price * quantity
                
            self.positions[asset] -= quantity
            self.current_capital += trade_value
        
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
        
        # 记录交易
        trade = {
            'timestamp': current_timestamp,
            'symbol': asset,
            'direction': action.upper(),
            'price': price,
            'quantity': int(quantity),
            'trigger_type': trigger_type,
            'grid_level': grid_level,
            'atr_value': self.current_atr.get(asset, 0) if hasattr(self, 'current_atr') else 0,
            'volatility_weight_ratio': f"{self.volatility_weights[self.asset_1]:.2f}:{self.volatility_weights[self.asset_2]:.2f}",
            'circuit_breaker_active': self.circuit_breaker_triggered,
            'pre_trade_cash': pre_trade_cash,
            'post_trade_cash': self.current_capital,
            'position_changes': json.dumps(position_changes),
            'reason': reason,
            'boil_price': self.current_prices.get(self.asset_1, 0),
            'kold_price': self.current_prices.get(self.asset_2, 0)
        }
        self.trades.append(trade)
        
        logger.info(f"执行交易: {action} {asset} {quantity:.4f}股 @ {price:.2f}, 价值: {trade_value:.2f}, 原因: {reason}")
    
    def check_grid_signals(self, data: pd.DataFrame) -> None:
        """
        检查网格交易信号
        
        Args:
            data: 包含价格数据的DataFrame
        """
        for asset in [self.asset_1, self.asset_2]:
            if len(data) < 2:
                continue
                
            current_price = data[f'{asset}_close'].iloc[-1]
            previous_price = data[f'{asset}_close'].iloc[-2]
            
            # 找到当前价格所在的网格区间
            for i in range(len(self.grids[asset]) - 1):
                lower_grid = self.grids[asset][i]
                upper_grid = self.grids[asset][i + 1]
                
                # 价格上穿网格线
                if previous_price < upper_grid <= current_price:
                    # 卖出信号
                    max_position_value = self.initial_capital * self.max_single_position_pct
                    current_position_value = self.positions[asset] * current_price
                    
                    # 计算可卖出的价值
                    sell_value = min(
                        max_position_value * 0.2,  # 最大仓位的20%
                        current_position_value * 0.2  # 当前仓位的20%
                    )
                    
                    if sell_value > 0:
                        quantity = sell_value / current_price
                        self.execute_trade(asset, 'sell', current_price, quantity, f"网格上穿: {lower_grid:.2f}-{upper_grid:.2f}")
                
                # 价格下穿网格线
                if previous_price > lower_grid >= current_price:
                    # 买入信号
                    max_position_value = self.initial_capital * self.max_single_position_pct
                    current_position_value = self.positions[asset] * current_price
                    
                    # 计算可买入的价值
                    available_position_value = max_position_value - current_position_value
                    buy_value = min(
                        available_position_value * 0.2,  # 可用仓位的20%
                        self.current_capital * 0.2  # 可用资金的20%
                    )
                    
                    if buy_value > 0:
                        quantity = buy_value / current_price
                        self.execute_trade(asset, 'buy', current_price, quantity, f"网格下穿: {lower_grid:.2f}-{upper_grid:.2f}")
    
    def rebalance_portfolio(self, data: pd.DataFrame) -> None:
        """
        再平衡投资组合
        
        Args:
            data: 包含价格数据的DataFrame
        """
        logger.info("执行投资组合再平衡")
        
        # 计算当前总价值
        total_value = self.calculate_portfolio_value(data)
        
        # 计算目标仓位
        target_value_per_asset = total_value * self.max_single_position_pct
        
        for asset in [self.asset_1, self.asset_2]:
            current_price = data[f'{asset}_close'].iloc[-1]
            current_position_value = self.positions[asset] * current_price
            
            value_difference = target_value_per_asset - current_position_value
            
            if abs(value_difference) / target_value_per_asset > 0.1:  # 偏离超过10%才调整
                if value_difference > 0:
                    # 需要买入
                    quantity = value_difference / current_price
                    self.execute_trade(asset, 'buy', current_price, quantity, "再平衡买入")
                else:
                    # 需要卖出
                    quantity = abs(value_difference) / current_price
                    self.execute_trade(asset, 'sell', current_price, quantity, "再平衡卖出")
    
    def hedge_adjustment(self, data: pd.DataFrame) -> None:
        """
        对冲调整
        
        Args:
            data: 包含价格数据的DataFrame
        """
        logger.info("执行对冲调整")
        
        # 计算当前总价值
        total_value = self.calculate_portfolio_value(data)
        
        # 计算每个资产的当前价值占比
        asset_1_price = data[f'{self.asset_1}_close'].iloc[-1]
        asset_2_price = data[f'{self.asset_2}_close'].iloc[-1]
        
        asset_1_value = self.positions[self.asset_1] * asset_1_price
        asset_2_value = self.positions[self.asset_2] * asset_2_price
        
        asset_1_weight = asset_1_value / total_value
        asset_2_weight = asset_2_value / total_value
        
        # 目标是使两个资产的权重接近平衡
        target_weight = self.max_single_position_pct
        
        # 调整资产1
        if asset_1_weight > target_weight * 1.2:  # 超过目标的20%
            # 卖出资产1
            excess_value = asset_1_value - (total_value * target_weight)
            quantity = excess_value / asset_1_price
            self.execute_trade(self.asset_1, 'sell', asset_1_price, quantity, "对冲调整卖出")
        elif asset_1_weight < target_weight * 0.8:  # 低于目标的20%
            # 买入资产1
            deficit_value = (total_value * target_weight) - asset_1_value
            quantity = deficit_value / asset_1_price
            self.execute_trade(self.asset_1, 'buy', asset_1_price, quantity, "对冲调整买入")
        
        # 调整资产2
        if asset_2_weight > target_weight * 1.2:  # 超过目标的20%
            # 卖出资产2
            excess_value = asset_2_value - (total_value * target_weight)
            quantity = excess_value / asset_2_price
            self.execute_trade(self.asset_2, 'sell', asset_2_price, quantity, "对冲调整卖出")
        elif asset_2_weight < target_weight * 0.8:  # 低于目标的20%
            # 买入资产2
            deficit_value = (total_value * target_weight) - asset_2_value
            quantity = deficit_value / asset_2_price
            self.execute_trade(self.asset_2, 'buy', asset_2_price, quantity, "对冲调整买入")
    
    def reset_circuit_breaker(self) -> None:
        """重置熔断状态"""
        self.circuit_breaker_triggered = False
        logger.info("熔断状态已重置")
    
    def establish_initial_position(self, data: pd.DataFrame) -> None:
        """
        执行初始建仓操作
        
        Args:
            data: 包含价格数据的DataFrame
        """
        logger.info("执行初始建仓")
        
        # 计算当前总价值
        total_value = self.initial_capital
        
        # 计算目标仓位
        target_value_per_asset = total_value * self.max_single_position_pct
        
        for asset in [self.asset_1, self.asset_2]:
            current_price = data[f'{asset}_close'].iloc[-1]
            
            # 计算购买数量
            quantity = target_value_per_asset / current_price
            
            # 执行交易，标记为初始建仓
            self.execute_trade(asset, 'buy', current_price, quantity, "初始建仓")
        
        # 标记初始建仓已完成
        self.initial_position_established = True
        self.last_rebalance_date = data.index[-1]  # 设置最后再平衡日期为当前日期
        
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
            return {"status": "circuit_breaker_triggered"}
            
        # 更新网格
        self.update_grid_levels(data)
        
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
            
        # 检查对冲偏离
        if self.check_hedge_deviation(data):
            self.hedge_adjustment(data)
            
        # 检查网格交易信号
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
            "cash": self.current_capital
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
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
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