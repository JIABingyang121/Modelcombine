# -*- coding: utf-8 -*-
"""
专门用于下载 AEMO NSW (New South Wales) 负荷数据的脚本
目标: 2023-2024年数据，小时级聚合
"""
import pandas as pd
import requests
import io
import time
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_aemo_hourly import aggregate_hourly

# 配置
REGION = "NSW1"
YEARS = [2023, 2024]
OUTPUT_DIR = "data/external/aemo_nsw"
OUTPUT_FILE = f"{OUTPUT_DIR}/load.csv"

# 伪装头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://aemo.com.au/'
}

def download_and_process():
    print(f"🚀 开始下载 AEMO {REGION} 数据 ({YEARS})...")
    
    # 1. 确保目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    all_frames = []

    for year in YEARS:
        for month in range(1, 13):
            # URL Pattern: PRICE_AND_DEMAND_YYYYMM_REGION.csv
            filename = f"PRICE_AND_DEMAND_{year}{month:02d}_{REGION}.csv"
            url = f"https://aemo.com.au/aemo/data/nem/priceanddemand/{filename}"
            
            print(f"  - Fetching {year}-{month:02d} ... ", end="")
            sys.stdout.flush()
            
            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                if resp.status_code == 200:
                    df = pd.read_csv(io.StringIO(resp.content.decode('utf-8')))
                    df.columns = [str(c).upper().strip() for c in df.columns]
                    df["source_file"] = filename
                    all_frames.append(df)
                    print("✅ OK")
                elif resp.status_code == 404:
                    print("❌ Not Found (未来数据?)")
                else:
                    print(f"⚠️ Failed ({resp.status_code})")
            except Exception as e:
                print(f"⚠️ Error: {e}")
            
            # 礼貌性延时
            time.sleep(0.5)

    if not all_frames:
        print("❌ 下载失败：没有获取到任何数据。请检查网络或 Region 代码。")
        return

    # 2. 合并与清洗（聚合统一走 scripts.prepare_aemo_hourly）
    print("\n🔄 正在合并与清洗...")
    full_df = pd.concat(all_frames, ignore_index=True)

    if 'SETTLEMENTDATE' in full_df.columns and 'TOTALDEMAND' in full_df.columns:
        hourly, _stats = aggregate_hourly(full_df, region=REGION)
        clean_df = hourly.rename(columns={"timestamp": "ds", "load": "y"})

        # 3. 保存
        clean_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ 成功！数据已保存至: {OUTPUT_FILE}")
        print(f"📊 记录数: {len(clean_df)}")
        print(f"📅 时间范围: {clean_df['ds'].min()} 到 {clean_df['ds'].max()}")
        print(clean_df.head())
    else:
        print("❌ 错误：关键列 SETTLEMENTDATE 或 TOTALDEMAND 缺失。")
        print("现有列:", full_df.columns.tolist())

if __name__ == "__main__":
    download_and_process()