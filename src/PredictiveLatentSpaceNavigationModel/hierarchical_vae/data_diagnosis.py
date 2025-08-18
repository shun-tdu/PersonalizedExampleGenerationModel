import numpy as np
import pandas as pd
import argparse
import sys
import os


def analyze_data(filepath):
    """データファイルを分析し、分散を確認する"""
    # 1. ファイルの存在確認
    if not os.path.exists(filepath):
        print(f"❌ エラー: ファイルが見つかりません: {filepath}")
        sys.exit(1)

    # 2. ファイルの読み込みとエラーハンドリング
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"❌ エラー: ファイルの読み込みに失敗しました: {e}")
        sys.exit(1)

    # 3. 必要なカラムの存在確認
    required_cols = ["subject_id", "HandlePosX", "HandlePosY"]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ エラー: ファイルに必要なカラム {required_cols} がありません。")
        sys.exit(1)

    # 4. 分析の実行
    print(f"✅ ファイル '{os.path.basename(filepath)}' を正常に読み込みました。")
    print(f"被験者数: {df['subject_id'].nunique()}")

    var_pos_x = df['HandlePosX'].var()
    var_pos_y = df['HandlePosY'].var()
    var_vel_x = df['HandleVelX'].var()
    var_vel_y = df['HandleVelY'].var()
    var_acc_x = df['HandleAccX'].var()
    var_acc_y = df['HandleAccY'].var()
    var_jerk_x = np.gradient(df['HandleAccX'], axis=0).var()
    var_jerk_y = np.gradient(df['HandleAccY'], axis=0).var()

    print(f"軌道の分散: X={var_pos_x:.6f}, Y={var_pos_y:.6f}")
    print(f"速度の分散: X={var_vel_x:.6f}, Y={var_vel_y:.6f}")
    print(f"加速度の分散: X={var_acc_x:.6f}, Y={var_acc_y:.6f}")
    print(f"ジャークの分散: X={var_jerk_x:.6f}, Y={var_jerk_y:.6f}")




if __name__ == "__main__":
    # コマンドライン引数を設定
    parser = argparse.ArgumentParser(description="Parquetデータファイルの分散を分析します。")
    parser.add_argument(
        "filepath",  # 引数名を "filepath" に変更
        type=str,
        help="分析対象の my_data.parquet ファイルへのパス"
    )
    args = parser.parse_args()

    # 分析関数を呼び出し
    analyze_data(args.filepath)