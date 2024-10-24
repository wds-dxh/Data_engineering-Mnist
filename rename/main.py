import os
import shutil

def rename_and_copy_images(base_dir, output_dir, start_group_number='09'):
    # 创建 output_subm 目录，并创建 0-9 的子目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(10):
        label_dir = os.path.join(output_dir, str(i))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

    # 获取所有人员的文件夹名称
    persons = [p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))]

    # 遍历每个人员的文件夹，成员编号递增
    member_number = 0

    for person in persons:
        person_dir = os.path.join(base_dir, person)

        # 检查是否存在完整的 0-9 文件夹结构
        expected_labels = [str(i) for i in range(10)]
        actual_labels = [l for l in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, l))]
        
        if not all(label in actual_labels for label in expected_labels):
            print(f"Skipping {person}: incomplete 0-9 folder structure")
            continue  # 跳过该用户文件夹

        # 成员编号（递增），每个人员对应一个编号
        member_number_str = f"{member_number:02d}"

        # 遍历每个标签文件夹（即：0-9）
        for label in expected_labels:
            label_dir = os.path.join(person_dir, label)

            # 获取当前标签文件夹下的所有图片文件
            images = [img for img in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, img))]

            # 按顺序对图片进行重命名并复制到对应的 output_subm 目录
            for i, image in enumerate(sorted(images)):
                # 排序号从 0 开始
                sort_number = f"{i:02d}"

                # 构建新的文件名：组号_成员编号_标签_排序.png
                new_name = f"{start_group_number}_{member_number_str}_{label}_{sort_number}.png"
                old_path = os.path.join(label_dir, image)
                new_path = os.path.join(output_dir, label, new_name)

                # 复制图片到 output_subm 对应的标签文件夹下并改为 png 格式
                shutil.copy(old_path, new_path)
                print(f"Copied and Renamed: {old_path} -> {new_path}")

        # 成员编号递增
        member_number += 1


# 调用函数，传入你的目录路径
base_directory = './'
output_directory = './output_subm'
rename_and_copy_images(base_directory, output_directory, start_group_number='09')
