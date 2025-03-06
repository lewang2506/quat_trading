import argparse
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

from data_processor import DataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"../logs/download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DownloadData')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='下载分钟级别历史数据')
    
    # 资产对
    parser.add_argument('--symbols', type=str, nargs='+', default=['BOIL', 'KOLD'],
                        help='股票代码列表，例如 BOIL KOLD')
    
    # 时间范围
    parser.add_argument('--start_date', type=str, default=None,
                        help='开始日期，格式为 YYYY-MM-DD，默认为两年前')
    parser.add_argument('--end_date', type=str, default=None,
                        help='结束日期，格式为 YYYY-MM-DD，默认为今天')
    
    # 数据间隔
    parser.add_argument('--interval', type=str, default='5m',
                        help='数据间隔，可选 1m, 5m, 15m, 30m, 60m')
    
    # 输出目录
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 确保日志目录存在
    os.makedirs("../logs", exist_ok=True)
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置默认时间范围
    if args.start_date is None:
        # 默认为两年前
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
        
    if args.end_date is None:
        # 默认为今天
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    logger.info(f"开始下载数据: {args.symbols}, 时间范围: {start_date} 至 {end_date}, 间隔: {args.interval}")
    
    # 创建数据处理器
    processor = DataProcessor(data_dir=args.data_dir)
    
    try:
        # 下载分钟级别数据
        data = processor.download_intraday_data(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval,
            save=True
        )
        
        # 计算数据统计信息
        logger.info(f"下载完成，数据形状: {data.shape}")
        
        # 计算相关性
        if len(args.symbols) >= 2:
            correlation = data[f'{args.symbols[0]}_close'].corr(data[f'{args.symbols[1]}_close'])
            logger.info(f"{args.symbols[0]} 和 {args.symbols[1]} 的相关性: {correlation:.4f}")
        
        # 显示数据样本
        logger.info("数据样本:")
        logger.info(data.head())
        
        # 保存数据信息
        info_file = os.path.join(args.data_dir, f"data_info_{args.interval}_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(info_file, 'w') as f:
            f.write(f"下载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"股票代码: {args.symbols}\n")
            f.write(f"时间范围: {start_date} 至 {end_date}\n")
            f.write(f"数据间隔: {args.interval}\n")
            f.write(f"数据形状: {data.shape}\n")
            f.write(f"数据列: {list(data.columns)}\n")
            f.write(f"数据时间范围: {data.index.min()} 至 {data.index.max()}\n")
            f.write(f"数据时间点数量: {len(data.index.unique())}\n")
            
            if len(args.symbols) >= 2:
                f.write(f"{args.symbols[0]} 和 {args.symbols[1]} 的相关性: {correlation:.4f}\n")
        
        logger.info(f"数据信息已保存至 {info_file}")
        
        return 0
    except Exception as e:
        logger.error(f"下载数据失败: {str(e)}")
        return 1

if __name__ == "__main__":
    main() 