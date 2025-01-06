
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from timeseries_algorithm.data.dataloader import load_csv, load_sample_data
import tempfile
import os

# テスト load_csv 関数
def test_load_csv():
    # 一時的なCSVファイルを作成
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        tmp.write("col1,col2,col3\n1,2,3\n4,5,6\n7,8,9")
        temp_name = tmp.name

    try:
        # load_csv 関数を呼び出す
        df = load_csv(temp_name)
        
        # 期待される DataFrame
        expected_df = pd.DataFrame({
            "col1": [1, 4, 7],
            "col2": [2, 5, 8],
            "col3": [3, 6, 9]
        })

        # DataFrame が一致することを確認
        assert_frame_equal(df.reset_index(drop=True), expected_df)
    finally:
        # 一時ファイルを削除
        os.remove(temp_name)

# 異常系テスト: load_csv 関数で存在しないファイルを指定
def test_load_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file.csv")

# 異常系テスト: load_csv 関数で空のファイルを読み込む
def test_load_csv_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        temp_name = tmp.name

    try:
        with pytest.raises(pd.errors.EmptyDataError):
            load_csv(temp_name)
    finally:
        os.remove(temp_name)

# 異常系テスト: load_csv 関数で不正なCSV形式を読み込む
def test_load_csv_invalid_format():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        tmp.write("col1,col2,col3\n1,2\n4,5,6,7")  # 不揃いな列
        temp_name = tmp.name

    try:
        with pytest.raises(pd.errors.ParserError):
            load_csv(temp_name)
    finally:
        os.remove(temp_name)

# テスト load_sample_data 関数
def test_load_sample_data():
    np.random.seed(0)  # 乱数シードを固定
    df = load_sample_data()
    
    # 期待される列
    assert "date" in df.columns
    assert "sales" in df.columns

    # 行数の確認
    assert len(df) == 100

    # 日付範囲の確認
    expected_start = pd.Timestamp('2022-01-01')
    expected_end = pd.Timestamp('2022-04-10')  # 2022-01-01 + 99日
    assert df['date'].iloc[0] == expected_start
    assert df['date'].iloc[-1] == expected_end

    # sales 列の値が 0 以上 100 未満であることを確認
    assert df['sales'].between(0, 100).all()
