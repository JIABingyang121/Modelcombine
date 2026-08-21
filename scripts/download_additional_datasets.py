import glob
import os
import shutil

import kagglehub


def download_smart_meter_london() -> None:
    print("\n" + "=" * 50)
    print("正在下载 Smart Meter Energy Consumption Data in London Households...")
    try:
        path = kagglehub.dataset_download("jeanmidev/smart-meters-in-london")
        print("数据集下载路径:", path)

        target_dir = os.path.join(os.getcwd(), "data", "external", "smart_meter_london")
        os.makedirs(target_dir, exist_ok=True)

        files = glob.glob(os.path.join(path, "*.csv"))
        for f in files:
            filename = os.path.basename(f)
            if "weather" in filename or "acorn" in filename:
                shutil.copy2(f, os.path.join(target_dir, filename))
                print(f"已复制 {filename}")

        block_files = glob.glob(os.path.join(path, "halfhourly_dataset", "halfhourly_dataset", "*.csv"))
        if not block_files:
            block_files = glob.glob(os.path.join(path, "halfhourly_dataset", "*.csv"))
        if not block_files:
            block_files = glob.glob(os.path.join(path, "block_*.csv"))

        for i, f in enumerate(block_files[:2]):
            filename = os.path.basename(f)
            shutil.copy2(f, os.path.join(target_dir, filename))
            print(f"已复制样本数据 {filename}")

        print(f"Smart Meter London 数据集处理完成。保存在: {target_dir}")
    except Exception as e:
        print(f"下载 Smart Meter London 失败: {e}")


def prepare_aemo_nsw_dir() -> None:
    print("\n" + "=" * 50)
    print("准备 AEMO NSW 数据目录...")
    target_dir = os.path.join(os.getcwd(), "data", "external", "aemo_nsw")
    os.makedirs(target_dir, exist_ok=True)

    readme_path = os.path.join(target_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("请将 AEMO NSW 原始数据放在此目录。\n")
        f.write("推荐文件名: load_2024.csv, load_2025.csv\n")
    print(f"已创建目录和说明文件: {target_dir}")


if __name__ == "__main__":
    os.makedirs(os.path.join(os.getcwd(), "data", "external"), exist_ok=True)
    download_smart_meter_london()
    prepare_aemo_nsw_dir()
