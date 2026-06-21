import re
import pandas as pd

# title_pattern = r'([^_]+)_grsna_abl_fo_(.+)'
title_pattern = r'([^_]+)_grsna_abl_combined_(.+)'
log_file_path = '/home/zhuyeqi/grsn-glad2/abl_combined_rw.out'
save_csv_path = log_file_path.replace('.out', '.csv')
save_excel_path = log_file_path.replace('.out', '.xlsx')

def parse_log_file(file_path, debug=False):
    """
    解析log文件，提取数据集、方法、AUC均值和标准差

    Args:
        file_path (str): log文件路径
        debug (bool): 是否显示调试信息

    Returns:
        pandas.DataFrame: 包含解析结果的DataFrame
    """

    # 存储结果的列表
    results = []

    # 正则表达式模式

    # 匹配Mean AUC行：Mean AUC: 78.8 (±1.1)
    mean_auc_pattern = r'Mean AUC:\s*([0-9.]+)\s*\(±([0-9.]+)\)'

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        with open(file_path, 'r', encoding='gbk') as file:
            content = file.read()

    # 按行分割内容
    lines = content.split('\n')

    current_dataset = None
    current_method = None

    for line in lines:
        line = line.strip()

        # 检查是否是标题行
        title_match = re.search(title_pattern, line)
        if title_match:
            current_dataset = title_match.group(1)  # 数据集名
            current_method = title_match.group(2)  # 方法名
            if debug:
                print(f"找到标题行: {line}")
                print(f"  数据集: {current_dataset}")
                print(f"  方法: {current_method}")
            continue

        # 检查是否是Mean AUC行
        if current_dataset and current_method:
            mean_auc_match = re.search(mean_auc_pattern, line)
            if mean_auc_match:
                auc_mean = float(mean_auc_match.group(1))
                auc_std = float(mean_auc_match.group(2))

                if debug:
                    print(f"找到Mean AUC行: {line}")
                    print(f"  AUC均值: {auc_mean}, 标准差: {auc_std}")

                results.append({
                    'Dataset': current_dataset,
                    'Method': current_method,
                    'AUC_Mean': auc_mean,
                    'AUC_Std': auc_std,
                    'AUC_string': f'{auc_mean} (±{auc_std})'
                })

                # 重置当前数据集和方法
                current_dataset = None
                current_method = None

    # 创建DataFrame
    df = pd.DataFrame(results)
    return df


def save_results(df, output_file=save_csv_path):
    """
    保存结果到CSV文件

    Args:
        df (pandas.DataFrame): 解析结果
        output_file (str): 输出文件路径
    """
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {output_file}")


def print_results(df):
    """
    打印结果

    Args:
        df (pandas.DataFrame): 解析结果
    """
    if df.empty:
        print("未找到匹配的数据")
        return

    print("解析结果：")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"数据集: {row['Dataset']}")
        print(f"方法: {row['Method']}")
        print(f"AUC: {row['AUC_Mean']:.1f} (±{row['AUC_Std']:.1f})")
        print("-" * 40)

    print(f"\n总共找到 {len(df)} 条记录")


# 主函数
def main():
    try:
        # 解析log文件（启用调试模式）
        print("开始解析文件，显示调试信息...")
        print("=" * 60)
        df = parse_log_file(log_file_path, debug=True)
        print("=" * 60)

        # 打印结果
        print_results(df)

        # 保存结果到CSV
        if not df.empty:
            save_results(df)

            # # 也可以保存为Excel格式
            # df.to_excel('extracted_results.xlsx', index=False)
            # print("结果也已保存到 extracted_results.xlsx")

    except FileNotFoundError:
        print(f"错误：找不到文件 {log_file_path}")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"解析过程中出现错误: {e}")


# 如果直接运行此脚本
if __name__ == "__main__":
    main()

# 你也可以直接调用函数来解析文件
# df = parse_log_file('your_log_file.txt')
# print_results(df)