import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, List, Dict, Optional
import os
import logging
from datetime import datetime, timedelta
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataProcessor')

class DataProcessor:
    """
    数据处理类，用于获取和处理历史数据
    """
    
    def __init__(self, data_dir: str = '../data'):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"数据处理器初始化完成，数据目录: {data_dir}")
    
    def download_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d',
        save: bool = True
    ) -> pd.DataFrame:
        """
        从Yahoo Finance下载历史数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式为 'YYYY-MM-DD'
            end_date: 结束日期，格式为 'YYYY-MM-DD'
            interval: 数据间隔，可选 '1d', '1h', '5m' 等
            save: 是否保存数据
            
        Returns:
            包含历史数据的DataFrame
        """
        logger.info(f"开始下载数据: {symbols}, 时间范围: {start_date} 至 {end_date}, 间隔: {interval}")
        
        # 下载数据
        data_dict = {}
        
        # 对于分钟级别数据，需要分段下载
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
            # Yahoo Finance对分钟级别数据有限制，通常只能下载最近7天的数据
            # 对于更长时间范围，需要分段下载
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 计算总天数
            total_days = (end_dt - start_dt).days
            
            # 如果总天数超过7天，需要分段下载
            if total_days > 7:
                logger.info(f"时间范围超过7天，将分段下载数据")
                
                # 分段下载，每段7天
                current_start = start_dt
                segment_data = {}
                
                while current_start < end_dt:
                    # 计算当前段的结束日期
                    current_end = min(current_start + timedelta(days=7), end_dt)
                    
                    logger.info(f"下载数据段: {current_start.strftime('%Y-%m-%d')} 至 {current_end.strftime('%Y-%m-%d')}")
                    
                    for symbol in symbols:
                        try:
                            ticker = yf.Ticker(symbol)
                            hist = ticker.history(
                                start=current_start.strftime('%Y-%m-%d'),
                                end=current_end.strftime('%Y-%m-%d'),
                                interval=interval
                            )
                            
                            if symbol not in segment_data:
                                segment_data[symbol] = []
                                
                            segment_data[symbol].append(hist)
                            logger.info(f"成功下载 {symbol} 数据段，共 {len(hist)} 条记录")
                            
                            # 避免频繁请求导致API限制
                            time.sleep(1)
                        except Exception as e:
                            logger.error(f"下载 {symbol} 数据段失败: {str(e)}")
                            continue
                    
                    # 更新当前开始日期
                    current_start = current_end
                
                # 合并各段数据
                for symbol, data_list in segment_data.items():
                    if data_list:
                        combined_data = pd.concat(data_list)
                        # 去除重复数据
                        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
                        # 重命名列
                        combined_data.columns = [f'{symbol}_{col.lower()}' for col in combined_data.columns]
                        data_dict[symbol] = combined_data
            else:
                # 时间范围在7天内，直接下载
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(start=start_date, end=end_date, interval=interval)
                        
                        # 重命名列
                        hist.columns = [f'{symbol}_{col.lower()}' for col in hist.columns]
                        
                        data_dict[symbol] = hist
                        logger.info(f"成功下载 {symbol} 数据，共 {len(hist)} 条记录")
                    except Exception as e:
                        logger.error(f"下载 {symbol} 数据失败: {str(e)}")
                        continue
        else:
            # 对于日线及以上级别数据，直接下载
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date, interval=interval)
                    
                    # 重命名列
                    hist.columns = [f'{symbol}_{col.lower()}' for col in hist.columns]
                    
                    data_dict[symbol] = hist
                    logger.info(f"成功下载 {symbol} 数据，共 {len(hist)} 条记录")
                except Exception as e:
                    logger.error(f"下载 {symbol} 数据失败: {str(e)}")
                    continue
        
        # 合并数据
        if not data_dict:
            raise ValueError("没有成功下载任何数据")
            
        # 使用第一个数据的索引作为基准
        base_index = list(data_dict.values())[0].index
        merged_data = pd.DataFrame(index=base_index)
        
        for symbol, data in data_dict.items():
            # 确保索引一致
            data = data.reindex(base_index)
            
            # 合并数据
            for col in data.columns:
                merged_data[col] = data[col]
        
        # 保存数据
        if save:
            interval_str = interval.replace('m', 'min').replace('h', 'hour').replace('d', 'day')
            file_name = f"historical_data_{interval_str}_{datetime.now().strftime('%Y%m%d')}.csv"
            file_path = os.path.join(self.data_dir, file_name)
            merged_data.to_csv(file_path)
            logger.info(f"数据已保存至 {file_path}")
        
        return merged_data
    
    def download_intraday_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '5m',
        save: bool = True
    ) -> pd.DataFrame:
        """
        从Yahoo Finance下载分钟级别历史数据，处理长时间范围的下载
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式为 'YYYY-MM-DD'
            end_date: 结束日期，格式为 'YYYY-MM-DD'
            interval: 数据间隔，可选 '1m', '5m', '15m', '30m', '60m'
            save: 是否保存数据
            
        Returns:
            包含历史数据的DataFrame
        """
        logger.info(f"开始下载分钟级别数据: {symbols}, 时间范围: {start_date} 至 {end_date}, 间隔: {interval}")
        
        # 转换日期
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 计算总天数
        total_days = (end_dt - start_dt).days
        
        # 分段下载，每段60天
        segment_size = 60  # Yahoo Finance对分钟级别数据的限制
        
        all_data = {}
        current_start = start_dt
        
        while current_start < end_dt:
            # 计算当前段的结束日期
            current_end = min(current_start + timedelta(days=segment_size), end_dt)
            
            logger.info(f"下载数据段: {current_start.strftime('%Y-%m-%d')} 至 {current_end.strftime('%Y-%m-%d')}")
            
            try:
                # 使用yfinance下载当前段的数据
                segment_data = yf.download(
                    tickers=symbols,
                    start=current_start.strftime('%Y-%m-%d'),
                    end=current_end.strftime('%Y-%m-%d'),
                    interval=interval,
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=False
                )
                
                # 处理多个股票的情况
                if len(symbols) > 1:
                    # 重组数据结构
                    for symbol in symbols:
                        if symbol not in all_data:
                            all_data[symbol] = []
                        
                        # 提取当前股票的数据
                        symbol_data = segment_data[symbol].copy()
                        
                        # 重命名列
                        symbol_data.columns = [f'{symbol}_{col.lower()}' for col in symbol_data.columns]
                        
                        all_data[symbol].append(symbol_data)
                else:
                    # 单个股票的情况
                    symbol = symbols[0]
                    if symbol not in all_data:
                        all_data[symbol] = []
                    
                    # 重命名列
                    segment_data.columns = [f'{symbol}_{col.lower()}' for col in segment_data.columns]
                    
                    all_data[symbol].append(segment_data)
                
                logger.info(f"成功下载数据段，共 {len(segment_data)} 条记录")
            except Exception as e:
                logger.error(f"下载数据段失败: {str(e)}")
            
            # 更新当前开始日期
            current_start = current_end
            
            # 避免频繁请求导致API限制
            time.sleep(2)
        
        # 合并各段数据
        merged_data = {}
        for symbol, data_list in all_data.items():
            if data_list:
                try:
                    # 合并该股票的所有数据段
                    symbol_data = pd.concat(data_list)
                    
                    # 去除重复数据
                    symbol_data = symbol_data[~symbol_data.index.duplicated(keep='first')]
                    
                    merged_data[symbol] = symbol_data
                    logger.info(f"成功合并 {symbol} 数据，共 {len(symbol_data)} 条记录")
                except Exception as e:
                    logger.error(f"合并 {symbol} 数据失败: {str(e)}")
        
        # 合并所有股票的数据
        if not merged_data:
            raise ValueError("没有成功下载任何数据")
        
        # 创建最终的DataFrame
        final_data = pd.DataFrame()
        
        # 合并所有股票的数据
        for symbol, data in merged_data.items():
            if final_data.empty:
                final_data = data
            else:
                # 合并数据，使用外连接确保保留所有时间点
                final_data = pd.merge(final_data, data, left_index=True, right_index=True, how='outer')
        
        # 保存数据
        if save and not final_data.empty:
            interval_str = interval.replace('m', 'min').replace('h', 'hour').replace('d', 'day')
            file_name = f"historical_data_{interval_str}_{datetime.now().strftime('%Y%m%d')}.csv"
            file_path = os.path.join(self.data_dir, file_name)
            final_data.to_csv(file_path)
            logger.info(f"数据已保存至 {file_path}")
        
        return final_data
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载历史数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            包含历史数据的DataFrame
        """
        logger.info(f"加载数据: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 读取数据
        data = pd.read_csv(file_path)
        
        # 确保日期列存在
        if 'date' in data.columns:
            # 转换日期列为日期类型
            data['date'] = pd.to_datetime(data['date'])
            
            # 设置日期为索引
            data.set_index('date', inplace=True)
        elif data.index.name == 'date':
            # 确保索引是日期类型
            data.index = pd.to_datetime(data.index)
        else:
            # 假设第一列是日期
            data.index = pd.to_datetime(data.iloc[:, 0])
            data = data.iloc[:, 1:]
            data.index.name = 'date'
        
        logger.info(f"成功加载数据，共 {len(data)} 条记录")
        
        return data
    
    def prepare_pair_data(
        self,
        data: pd.DataFrame,
        asset_pair: Tuple[str, str],
        correlation_window: int = 60
    ) -> Tuple[pd.DataFrame, float]:
        """
        准备资产对数据，并计算相关性
        
        Args:
            data: 原始数据
            asset_pair: 资产对
            correlation_window: 计算相关性的窗口大小
            
        Returns:
            处理后的数据和相关性系数
        """
        asset_1, asset_2 = asset_pair
        
        # 检查数据是否包含所需的列
        required_columns = []
        for asset in asset_pair:
            required_columns.extend([f'{asset}_open', f'{asset}_high', f'{asset}_low', f'{asset}_close', f'{asset}_volume'])
            
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少以下列: {missing_columns}")
        
        # 计算相关性
        if len(data) >= correlation_window:
            # 使用收盘价计算相关性
            correlation = data[f'{asset_1}_close'].tail(correlation_window).corr(data[f'{asset_2}_close'].tail(correlation_window))
        else:
            correlation = data[f'{asset_1}_close'].corr(data[f'{asset_2}_close'])
            
        logger.info(f"{asset_1} 和 {asset_2} 的相关性: {correlation:.4f}")
        
        return data, correlation
    
    def generate_sample_data(
        self,
        asset_pair: Tuple[str, str],
        days: int = 252,
        volatility: float = 0.02,
        correlation: float = -0.8,
        trend: float = 0.0001,
        save: bool = True
    ) -> pd.DataFrame:
        """
        生成样本数据，用于测试
        
        Args:
            asset_pair: 资产对
            days: 天数
            volatility: 波动率
            correlation: 相关性
            trend: 趋势
            save: 是否保存数据
            
        Returns:
            生成的样本数据
        """
        asset_1, asset_2 = asset_pair
        
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' 表示工作日
        
        # 生成价格数据
        np.random.seed(42)  # 设置随机种子，确保结果可重现
        
        # 生成相关的随机数
        rho = correlation
        size = (len(date_range), 2)
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        
        # 生成相关的随机游走
        random_walk = np.random.multivariate_normal(mean, cov, size=len(date_range))
        
        # 生成价格序列
        price_1 = 100.0  # 初始价格
        price_2 = 100.0  # 初始价格
        
        prices_1 = [price_1]
        prices_2 = [price_2]
        
        for i in range(1, len(date_range)):
            # 添加趋势和波动
            price_1 = price_1 * (1 + trend + volatility * random_walk[i, 0])
            price_2 = price_2 * (1 + trend + volatility * random_walk[i, 1])
            
            prices_1.append(price_1)
            prices_2.append(price_2)
        
        # 创建DataFrame
        data = pd.DataFrame(index=date_range)
        
        # 添加OHLC数据
        for i, asset in enumerate([asset_1, asset_2]):
            prices = prices_1 if i == 0 else prices_2
            
            # 生成开盘价、最高价、最低价
            opens = prices.copy()
            highs = [p * (1 + np.random.uniform(0, 0.01)) for p in prices]
            lows = [p * (1 - np.random.uniform(0, 0.01)) for p in prices]
            volumes = [np.random.randint(100000, 1000000) for _ in prices]
            
            data[f'{asset}_open'] = opens
            data[f'{asset}_high'] = highs
            data[f'{asset}_low'] = lows
            data[f'{asset}_close'] = prices
            data[f'{asset}_volume'] = volumes
        
        # 保存数据
        if save:
            file_name = f"sample_data_{datetime.now().strftime('%Y%m%d')}.csv"
            file_path = os.path.join(self.data_dir, file_name)
            data.to_csv(file_path)
            logger.info(f"样本数据已保存至 {file_path}")
        
        return data


if __name__ == "__main__":
    # 示例用法
    processor = DataProcessor()
    
    # 生成样本数据
    asset_pair = ('BOIL', 'KOLD')
    sample_data = processor.generate_sample_data(
        asset_pair=asset_pair,
        days=252,  # 约一年的交易日
        volatility=0.02,
        correlation=-0.8,  # 负相关
        trend=0.0001,
        save=True
    )
    
    print(f"生成的样本数据: {sample_data.shape}")
    print(sample_data.head())
    
    # 计算相关性
    correlation = sample_data['BOIL_close'].corr(sample_data['KOLD_close'])
    print(f"BOIL 和 KOLD 的相关性: {correlation:.4f}") 