"""Database connection and session management for MySQL."""

import logging
from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings
from services.database.models import Base

logger = logging.getLogger("memorygraph.database")

_engine = None
_SessionFactory = None


def get_engine():
    """Return singleton SQLAlchemy engine if MySQL is enabled."""
    global _engine, _SessionFactory
    settings = get_settings()

    if not settings.MYSQL_ENABLED:
        return None

    if _engine is None:
        try:
            _engine = create_engine(
                settings.MYSQL_URL,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10,
                max_overflow=20,
            )
            _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
            logger.info("Initialized MySQL engine for %s", settings.MYSQL_URL.split("@")[-1])
        except Exception as e:
            logger.error("Failed to initialize MySQL engine: %s", str(e))
            _engine = None
            _SessionFactory = None

    return _engine


def init_db() -> bool:
    """Create all registered database tables if MySQL is enabled."""
    settings = get_settings()
    if not settings.MYSQL_ENABLED:
        logger.info("MySQL is disabled in settings. Skipping table initialization.")
        return False

    engine = get_engine()
    if not engine:
        return False

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("MySQL tables initialized successfully.")
        return True
    except Exception as e:
        logger.error("Failed to initialize MySQL database tables: %s", str(e))
        return False


def get_db_session() -> Generator[Optional[Session], None, None]:
    """Provide a transactional database session scope."""
    get_engine()
    global _SessionFactory
    if _SessionFactory is None:
        yield None
        return

    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def is_database_connected() -> bool:
    """Check if MySQL connection is live and healthy."""
    settings = get_settings()
    if not settings.MYSQL_ENABLED:
        return False

    engine = get_engine()
    if not engine:
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
