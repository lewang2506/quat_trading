import argparse
import os
import logging
import pandas as pd
from datetime import datetime
import json

from strategy import PairGridStrategy
from backtest import Backtest, prepare_data
from data_processor import DataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"../logs/strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Main')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='负相关资产对网格交易策略')
    
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['backtest', 'generate_data', 'download_data'], 
                        default='backtest', help='运行模式')
    
    # 资产对
    parser.add_argument('--asset_pair', type=str, nargs=2, default=['BOIL', 'KOLD'],
                        help='交易的资产对，例如 BOIL KOLD')
    
    # 回测参数
    parser.add_argument('--data_file', type=str, help='数据文件路径')
    parser.add_argument('--start_date', type=str, help='回测开始日期，格式为 YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='回测结束日期，格式为 YYYY-MM-DD')
    parser.add_argument('--initial_capital', type=float, default=100000, help='初始资金')
    
    # 策略参数
    parser.add_argument('--atr_period', type=int, default=14, help='ATR周期')
    parser.add_argument('--grid_distance_pct', type=float, default=0.5, help='网格间距为ATR的百分比')
    parser.add_argument('--max_single_position_pct', type=float, default=0.2, help='最大单边仓位占总资金的百分比')
    parser.add_argument('--hedge_deviation_threshold', type=float, default=0.05, help='对冲偏离阈值')
    parser.add_argument('--rebalance_period', type=str, default='W-FRI', help='再平衡周期')
    parser.add_argument('--circuit_breaker_threshold', type=float, default=0.15, help='熔断阈值')
    
    # 数据生成参数
    parser.add_argument('--days', type=int, default=252, help='生成数据的天数')
    parser.add_argument('--volatility', type=float, default=0.02, help='波动率')
    parser.add_argument('--correlation', type=float, default=-0.8, help='相关性')
    parser.add_argument('--trend', type=float, default=0.0001, help='趋势')
    
    # 数据下载参数
    parser.add_argument('--download_start_date', type=str, help='下载数据的开始日期，格式为 YYYY-MM-DD')
    parser.add_argument('--download_end_date', type=str, help='下载数据的结束日期，格式为 YYYY-MM-DD')
    parser.add_argument('--interval', type=str, default='1d', help='数据间隔，可选 1d, 1wk, 1mo 等')
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default='../backtest', help='输出目录')
    parser.add_argument('--data_dir', type=str, default='../data', help='数据目录')
    
    # 配置文件
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(args, output_dir):
    """保存配置"""
    config = vars(args)
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"配置已保存至 {config_path}")

def run_backtest(args):
    """运行回测"""
    logger.info("开始回测模式")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    save_config(args, args.output_dir)
    
    # 创建策略实例
    strategy = PairGridStrategy(
        asset_pair=tuple(args.asset_pair),
        initial_capital=args.initial_capital,
        atr_period=args.atr_period,
        grid_distance_pct=args.grid_distance_pct,
        max_single_position_pct=args.max_single_position_pct,
        hedge_deviation_threshold=args.hedge_deviation_threshold,
        rebalance_period=args.rebalance_period,
        circuit_breaker_threshold=args.circuit_breaker_threshold
    )
    
    # 准备数据
    data = prepare_data(args.data_file, tuple(args.asset_pair))
    
    # 创建回测实例
    backtest = Backtest(
        strategy=strategy,
        data=data,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir
    )
    
    # 运行回测
    performance = backtest.run()
    
    # 打印性能指标
    logger.info("\n性能指标:")
    logger.info(f"总收益率: {performance['total_return']:.2%}")
    logger.info(f"年化收益率: {performance['annualized_return']:.2%}")
    logger.info(f"年化波动率: {performance['annualized_volatility']:.2%}")
    logger.info(f"夏普比率: {performance['sharpe_ratio']:.2f}")
    logger.info(f"最大回撤: {performance['max_drawdown']:.2%}")
    logger.info(f"交易次数: {performance['trade_count']}")
    
    return performance

def generate_sample_data(args):
    """生成样本数据"""
    logger.info("开始生成样本数据")
    
    # 创建数据处理器
    processor = DataProcessor(data_dir=args.data_dir)
    
    # 生成样本数据
    sample_data = processor.generate_sample_data(
        asset_pair=tuple(args.asset_pair),
        days=args.days,
        volatility=args.volatility,
        correlation=args.correlation,
        trend=args.trend,
        save=True
    )
    
    logger.info(f"生成的样本数据: {sample_data.shape}")
    
    # 计算相关性
    correlation = sample_data[f'{args.asset_pair[0]}_close'].corr(sample_data[f'{args.asset_pair[1]}_close'])
    logger.info(f"{args.asset_pair[0]} 和 {args.asset_pair[1]} 的相关性: {correlation:.4f}")
    
    # 返回生成的数据文件路径
    file_name = f"sample_data_{datetime.now().strftime('%Y%m%d')}.csv"
    file_path = os.path.join(args.data_dir, file_name)
    
    return file_path

def download_data(args):
    """下载历史数据"""
    logger.info("开始下载历史数据")
    
    # 创建数据处理器
    processor = DataProcessor(data_dir=args.data_dir)
    
    # 下载数据
    data = processor.download_data(
        symbols=args.asset_pair,
        start_date=args.download_start_date,
        end_date=args.download_end_date,
        interval=args.interval,
        save=True
    )
    
    logger.info(f"下载的数据: {data.shape}")
    
    # 计算相关性
    correlation = data[f'{args.asset_pair[0]}_close'].corr(data[f'{args.asset_pair[1]}_close'])
    logger.info(f"{args.asset_pair[0]} 和 {args.asset_pair[1]} 的相关性: {correlation:.4f}")
    
    # 返回下载的数据文件路径
    file_name = f"historical_data_{datetime.now().strftime('%Y%m%d')}.csv"
    file_path = os.path.join(args.data_dir, file_name)
    
    return file_path

def main():
    """主函数"""
    # 确保日志目录存在
    os.makedirs("../logs", exist_ok=True)
    
    # 解析命令行参数
    args = parse_args()
    
    # 如果提供了配置文件，从配置文件加载参数
    if args.config:
        config = load_config(args.config)
        # 更新参数
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    logger.info(f"运行模式: {args.mode}")
    
    if args.mode == 'generate_data':
        # 生成样本数据
        data_file = generate_sample_data(args)
        
        # 更新数据文件路径
        args.data_file = data_file
        
        # 如果需要，继续运行回测
        if input("是否继续运行回测? (y/n): ").lower() == 'y':
            args.mode = 'backtest'
            run_backtest(args)
    
    elif args.mode == 'download_data':
        # 下载历史数据
        data_file = download_data(args)
        
        # 更新数据文件路径
        args.data_file = data_file
        
        # 如果需要，继续运行回测
        if input("是否继续运行回测? (y/n): ").lower() == 'y':
            args.mode = 'backtest'
            run_backtest(args)
    
    elif args.mode == 'backtest':
        # 运行回测
        if not args.data_file:
            logger.error("回测模式需要提供数据文件路径")
            return
        
        run_backtest(args)

if __name__ == "__main__":
    main() 