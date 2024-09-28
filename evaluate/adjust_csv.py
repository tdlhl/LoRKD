import csv

def merge_and_write_csv(input_csv_path, output_csv_path):
    with open(input_csv_path, mode='r', encoding='utf-8') as input_file:
        csv_reader = csv.reader(input_file)
        rows = list(csv_reader)
    
    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        
        for row in rows:
            merged_column = f'"{row[0]}, {row[1]}"'
            new_row = [merged_column] + row[2:]
            csv_writer.writerow(new_row)
            
# 调用merge_and_write_csv函数
input_csv_path = 'nnunet_dataset_results_detailed.csv'  # 输入CSV文件路径
output_csv_path = 'nnunet_dataset_results_detailed_a.csv'  # 输出CSV文件路径
merge_and_write_csv(input_csv_path, output_csv_path)
