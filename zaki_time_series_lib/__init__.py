
from zaki_time_series_lib.config import settings
from zaki_time_series_lib.utils.logger import get_logger

logger = get_logger(__name__)
logger.info(f"Initializing zaki_time_series_lib v{settings.VERSION}")
