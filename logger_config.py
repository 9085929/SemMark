import os
import sys
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = 'my_app', log_file: str = 'logs/app.log', level: int = logging.INFO):
    """
    配置一个可复用的 logger。

    参数:
        name (str): logger 名称
        log_file (str): 日志文件路径
        level (logging.level): 日志等级
    返回:
        logging.Logger: 配置好的 logger 实例
    """
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复日志

    # 如果已经添加过 handler，就不重复添加
    if not logger.handlers:
        # 创建 formatter
        formatter = logging.Formatter(
            '[%(asctime)s][%(name)s][%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件 handler（支持轮转）
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
