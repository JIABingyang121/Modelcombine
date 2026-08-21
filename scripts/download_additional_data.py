import os
import shutil

import kagglehub


def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_smart_meters_london() -> None:
    print("\n" + "=" * 50)
    print("正在处理 Smart Meters in London 数据集...")

    target_dir = os.path.join(os.getcwd(), "data", "external", "london_smart_meters")
    ensure_dir(target_dir)

    local_archive_dir = os.path.join(os.getcwd(), "data", "archive")
    if os.path.exists(local_archive_dir) and os.listdir(local_archive_dir):
        print(f"检测到本地数据目录: {local_archive_dir}")
        print(f"正在导入到: {target_dir}")
        try:
            for root, _, files in os.walk(local_archive_dir):
                for file in files:
                    src_file = os.path.join(root, file)
                    rel_path = os.path.relpath(root, local_archive_dir)
                    dst_subdir = target_dir if rel_path == "." else os.path.join(target_dir, rel_path)
                    ensure_dir(dst_subdir)
                    dst_file = os.path.join(dst_subdir, file)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
            print("本地数据已成功导入。")
            return
        except Exception as e:
            print(f"导入本地数据时出错: {e}")
            print("尝试从 Kaggle 下载...")

    try:
        print("尝试从 Kaggle 下载...")
        path = kagglehub.dataset_download("jeanmidev/smart-meters-in-london")
        print(f"下载完成，缓存路径: {path}")
        print(f"正在复制文件到: {target_dir}")
        for filename in os.listdir(path):
            src = os.path.join(path, filename)
            dst = os.path.join(target_dir, filename)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)
        print("Smart Meters in London 数据集准备就绪。")
    except Exception as e:
        print(f"下载 Smart Meters in London 失败: {e}")


def prepare_aemo_nsw_dir() -> None:
    print("\n" + "=" * 50)
    print("准备 AEMO NSW 数据目录...")
    target_dir = os.path.join(os.getcwd(), "data", "external", "aemo_nsw")
    ensure_dir(target_dir)
    readme_path = os.path.join(target_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("请将 AEMO NSW 的原始负荷数据放置于本目录。\n")
        f.write("推荐文件名: load_2024.csv, load_2025.csv\n")
    print(f"AEMO NSW 目录已准备: {target_dir}")


if __name__ == "__main__":
    ensure_dir(os.path.join(os.getcwd(), "data", "external"))
    download_smart_meters_london()
    prepare_aemo_nsw_dir()
