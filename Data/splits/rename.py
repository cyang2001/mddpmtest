import pandas as pd

def add_columns_to_csv(input_csv_path, output_csv_path, setname_value, settype_value):
    # 读取CSV文件
    df = pd.read_csv(input_csv_path)

    # 插入新的列
    df['setname'] = setname_value
    df['settype'] = settype_value

    # 保存修改后的CSV文件
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")

# 示例使用
input_csv_path = 'Brats21_val.csv'  # 输入CSV文件路径
output_csv_path = 'Brats21_val.csv'  # 输出CSV文件路径
setname_value = 'Brats21'  # 你想要插入的setname的值
settype_value = 'val'  # 你想要插入的settype的值

add_columns_to_csv(input_csv_path, output_csv_path, setname_value, settype_value)