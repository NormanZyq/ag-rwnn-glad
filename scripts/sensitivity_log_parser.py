import re
import pandas as pd
from typing import List, Dict, Tuple, Optional

def parse_experiment_log(log_file_path: str) -> pd.DataFrame:
    """
    解析实验日志文件，提取数据集名称、偏差强度值和相应的AUC、F1结果

    Args:
        log_file_path: 日志文件路径

    Returns:
        包含解析结果的DataFrame
    """

    # 读取日志文件
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # 如果UTF-8编码失败，尝试其他编码
        with open(log_file_path, 'r', encoding='gbk') as file:
            content = file.read()

    # 定义正则表达式模式
    # 匹配实验标识行：{dataset name}_bias_strength_{value}
    experiment_pattern = r'(\w+)_bias_strength_([0-9.]+)'

    # 匹配结果行：Mean AUC: 99.5 (±0.3), Mean F1: 98.7 (±0.5)
    result_pattern = r'Mean AUC:\s*([0-9.]+)\s*\(±([0-9.]+)\),\s*Mean F1:\s*([0-9.]+)\s*\(±([0-9.]+)\)'

    # 存储解析结果
    results = []

    # 按行分割内容
    lines = content.split('\n')

    current_experiment = None

    for i, line in enumerate(lines):
        line = line.strip()

        # 检查是否是实验标识行
        exp_match = re.search(experiment_pattern, line)
        if exp_match:
            dataset_name = exp_match.group(1)
            bias_strength = float(exp_match.group(2))
            current_experiment = (dataset_name, bias_strength)
            continue

        # 检查是否是结果行
        result_match = re.search(result_pattern, line)
        if result_match and current_experiment is not None:
            auc_mean = float(result_match.group(1))
            auc_std = float(result_match.group(2))
            f1_mean = float(result_match.group(3))
            f1_std = float(result_match.group(4))

            # 添加到结果列表
            results.append({
                'Dataset': current_experiment[0],
                'Bias_Strength': current_experiment[1],
                'AUC_Mean': auc_mean,
                'AUC_Std': auc_std,
                'F1_Mean': f1_mean,
                'F1_Std': f1_std
            })

            # 重置当前实验（防止重复匹配）
            current_experiment = None

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 按数据集名称和偏差强度排序
    if not df.empty:
        df = df.sort_values(['Dataset', 'Bias_Strength']).reset_index(drop=True)

    return df

def save_results(df: pd.DataFrame, output_path: str = 'experiment_results.csv'):
    """
    保存结果到CSV文件

    Args:
        df: 包含结果的DataFrame
        output_path: 输出文件路径
    """
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"结果已保存到: {output_path}")

def print_summary(df: pd.DataFrame):
    """
    打印结果摘要

    Args:
        df: 包含结果的DataFrame
    """
    if df.empty:
        print("未找到任何实验结果")
        return

    print(f"\n解析到 {len(df)} 个实验结果")
    print(f"数据集数量: {df['Dataset'].nunique()}")
    print(f"数据集列表: {', '.join(df['Dataset'].unique())}")
    print(f"偏差强度值范围: {df['Bias_Strength'].min()} - {df['Bias_Strength'].max()}")

    print("\n实验结果表格:")
    print(df.to_string(index=False))

def main():
    """
    主函数：使用示例
    """
    # 修改这里的文件路径为你的实际日志文件路径
    log_file_path = '/home/zhuyeqi/grsn-glad2/sens0913.out'  # 替换为你的日志文件路径

    try:
        # 解析日志文件
        results_df = parse_experiment_log(log_file_path)

        # 打印摘要
        print_summary(results_df)

        # 保存结果
        if not results_df.empty:
            save_results(results_df, 'parsed_sens0913.csv')

            # 也可以保存为Excel格式
            results_df.to_excel('parsed_sens0913.xlsx', index=False)
            print("结果也已保存为Excel格式: parsed_sens0913.xlsx")

    except FileNotFoundError:
        print(f"错误：找不到文件 {log_file_path}")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"解析过程中出现错误: {str(e)}")

# 额外的实用函数
def filter_by_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    根据数据集名称筛选结果
    """
    return df[df['Dataset'] == dataset_name].copy()

def get_best_results(df: pd.DataFrame, metric: str = 'AUC_Mean') -> pd.DataFrame:
    """
    获取每个数据集的最佳结果

    Args:
        df: 结果DataFrame
        metric: 用于比较的指标列名 ('AUC_Mean' 或 'F1_Mean')

    Returns:
        包含每个数据集最佳结果的DataFrame
    """
    if df.empty:
        return df

    best_results = df.loc[df.groupby('Dataset')[metric].idxmax()]
    return best_results.reset_index(drop=True)

if __name__ == "__main__":
    main()