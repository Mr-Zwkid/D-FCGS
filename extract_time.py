import re
from statistics import mean

def extract_times_from_log(filename):
    compression_times = []
    decompression_times = []
    
    with open(filename, 'r') as f:
        for line in f:
            # 匹配压缩时间
            comp_match = re.search(r'Compression time: (\d+\.\d+)', line)
            if comp_match:
                compression_times.append(float(comp_match.group(1)))
            
            # 匹配解压时间
            decomp_match = re.search(r'Decompression time: (\d+\.\d+)', line)
            if decomp_match:
                decompression_times.append(float(decomp_match.group(1)))
    
    return compression_times, decompression_times

def print_statistics(times, operation):
    if times:
        avg_time = mean(times)
        print(f"\n{operation} 统计:")
        print(f"样本数: {len(times)}")
        print(f"平均时间: {avg_time:.4f}")
        print(f"最小时间: {min(times):.4f}")
        print(f"最大时间: {max(times):.4f}")
    else:
        print(f"\n没有找到 {operation} 时间数据")

def main():
    log_file = "/SSD2/chenzx/Projects/FCGS/rebuttal/0616-0.005-gof100/train_20250616_195601.log" 
    
    try:
        comp_times, decomp_times = extract_times_from_log(log_file)
        
        print_statistics(comp_times, "压缩")
        print_statistics(decomp_times, "解压")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{log_file}'")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()