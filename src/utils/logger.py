import logging
import sys

def get_production_logger(module_name: str) -> logging.Logger:
    """Provides a unified log formatting style across all pipeline stages."""
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] (%(name)s) : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger