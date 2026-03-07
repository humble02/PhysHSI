import torch
import os

def walk_and_inspect_pt(parent_folder):
    if not os.path.exists(parent_folder):
        print(f"❌ 错误: 父文件夹 '{parent_folder}' 不存在")
        return

    print(f"📂 正在深度遍历父目录: {parent_folder}")
    print("-" * 50)

    all_stats = {} # 用于存储 {文件夹名: {文件名: 长度}}

    # os.walk 会递归进入所有子文件夹
    for root, dirs, files in os.walk(parent_folder):
        # 过滤出当前目录下的 .pt 文件
        pt_files = [f for f in files if f.endswith('.pt')]
        
        if not pt_files:
            continue

        current_dir_name = os.path.relpath(root, parent_folder)
        print(f"📁 进入子目录: {current_dir_name} (发现 {len(pt_files)} 个文件)")
        
        dir_results = {}
        
        for file_name in sorted(pt_files):
            file_path = os.path.join(root, file_name)
            try:
                # 仅加载到 CPU
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                
                if 'base_height' in data:
                    # 获取 base_height 的第一维长度
                    seq_len = data['base_height'].shape[0]
                    dir_results[file_name] = seq_len
                else:
                    print(f"  ⚠️ {file_name}: 缺少 'base_height' 键")
            except Exception as e:
                print(f"  ❌ {file_name} 读取出错: {e}")
        
        if dir_results:
            all_stats[current_dir_name] = dir_results
            # 打印当前文件夹的简单汇总
            lens = list(dir_results.values())
            print(f"  ✅ 完成统计: 平均长度 {sum(lens)/len(lens):.1f}, 数量 {len(lens)}")

    return all_stats

if __name__ == "__main__":
    # 替换为你最外层的父文件夹路径
    PARENT_DIR = "/home/yaobicheng/boxmove/PhysHSI/legged_gym/resources/dataset/dataset_carrybox" 
    
    results = walk_and_inspect_pt(PARENT_DIR)

    # --- 最终导出示例 ---
    if results:
        print("\n" + "="*50)
        print("🚀 所有文件夹处理完毕！")
        # 统计总文件数
        total_files = sum(len(f_dict) for f_dict in results.values())
        print(f"统计总计: {len(results)} 个子文件夹，共 {total_files} 个有效文件。")