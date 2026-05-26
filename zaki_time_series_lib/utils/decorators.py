import functools
import time
import warnings
from typing import Any, Callable, Optional

from zaki_time_series_lib.utils.logger import get_logger


def log_entry_exit(logger: Optional[Any] = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            cls_name = ""
            if args and hasattr(args[0], '__class__'):
                cls_name = args[0].__class__.__name__ + "."
            func_name = f"{cls_name}{func.__name__}"
            logger.info(f">> ENTER: {func_name}")
            logger.debug(f"  args: {args[1:] if cls_name else args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"<< EXIT: {func_name}")
                return result
            except Exception as e:
                logger.error(f"<< EXIT (ERROR): {func_name} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator


def timer(logger: Optional[Any] = None, label: Optional[str] = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            cls_name = ""
            if args and hasattr(args[0], '__class__'):
                cls_name = args[0].__class__.__name__ + "."
            func_name = label or f"{cls_name}{func.__name__}"
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"[TIMER] {func_name} completed in {elapsed:.4f}s ({elapsed/60:.2f}min)")
            return result
        return wrapper
    return decorator


def deprecated(replacement: Optional[str] = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__} is deprecated."
            if replacement:
                msg += f" Use {replacement} instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(expected_type: type):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, arg in enumerate(args[1:], 1):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument {i} of {func.__name__} expected {expected_type.__name__}, got {type(arg).__name__}")
            for k, v in kwargs.items():
                if not isinstance(v, expected_type):
                    raise TypeError(f"Argument '{k}' of {func.__name__} expected {expected_type.__name__}, got {type(v).__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TimerContext:
    def __init__(self, name: str, logger: Optional[Any] = None):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        self.logger.info(f"[TIMER] {self.name} started...")
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.logger.info(f"[TIMER] {self.name} finished in {elapsed:.4f}s ({elapsed/60:.2f}min)")
