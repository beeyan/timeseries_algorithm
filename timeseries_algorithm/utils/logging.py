# logging.py

import logging
import sys

def setup_logger(name: str, log_level=logging.INFO):
    """
    ロガーを設定し、フォーマットや出力先を管理する例。
    他のモジュールから利用する際:
        logger = setup_logger(__name__, logging.DEBUG)
        logger.info("Log message")
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # すでにハンドラがある場合は追加しないようにする
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
