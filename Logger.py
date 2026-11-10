import logging

def get_logger(name):
    """
    创建日志处理器
    Args:
        name: 文件名

    Returns:
        返回日志处理器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    filename = "./logs_baseline/" + name
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(filename,encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger