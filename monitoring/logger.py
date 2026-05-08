import logging
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_10MB = 10 * 1024 * 1024


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "agent": record.name,
            "msg": record.getMessage(),
        }
        for key in ("symbol", "signal_id", "trade_id", "strategy"):
            if hasattr(record, key):
                obj[key] = getattr(record, key)
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False

    # Console handler (human-readable)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

    # Rotating file handler — 10 MB × 5 backups
    fh = RotatingFileHandler(LOG_DIR / "trading.log", maxBytes=_10MB, backupCount=5, encoding="utf-8")
    fh.setFormatter(_JsonFormatter())
    logger.addHandler(fh)

    # Separate audit log (also rotating, never propagates)
    audit_logger = logging.getLogger("audit")
    if not audit_logger.handlers:
        audit = RotatingFileHandler(LOG_DIR / "audit.log", maxBytes=_10MB, backupCount=10, encoding="utf-8")
        audit.setFormatter(_JsonFormatter())
        audit.setLevel(logging.INFO)
        audit_logger.addHandler(audit)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False

    return logger
