import pandas as pd

def load_csv(filepath: str) -> pd.DataFrame:
    """
    CSVファイルを読み込んで DataFrame を返す。
    ファイルパスのみ指定すれば簡単に使えるユーティリティ関数の例。
    """
    df = pd.read_csv(filepath)
    return df

def load_sample_data() -> pd.DataFrame:
    """
    例として、サンプルの時系列データを生成して返す関数。
    実際には外部APIや他の形式のデータソースから読み込む処理を書くこともできる。
    """
    import numpy as np
    import pandas as pd

    dates = pd.date_range(start='2021-01-01', periods=730, freq='D')
    values = np.random.rand(730) * 100
    data = pd.DataFrame({"date": dates, "sales": values})
    return data