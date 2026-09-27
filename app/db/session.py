from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from app.core.config import settings

DATABASE_URL = f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

engine = create_engine(
    DATABASE_URL
    , echo            = False
    # Connection pool settings
    , poolclass       = QueuePool
    , pool_pre_ping   = True        
    , pool_size       = 15
    , max_overflow    = 25
    , pool_timeout    = 60
    , pool_recycle    = 1800
    , pool_use_lifo   = True 

    # PostgreSQL specific connection args
    , connect_args = {
        "connect_timeout": 30
        , "keepalives": 1
        , "keepalives_idle": 60
        , "keepalives_interval": 10
        , "keepalives_count": 5
    }
)

SessionLocal = sessionmaker(
    autocommit = False
    , autoflush= False
    , bind     = engine
)

def get_session():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def check_connection() -> tuple[bool, str]:
    """Run SELECT 1 against Postgres. Returns (ok, detail)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    except Exception as e:
        return False, f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME} - {e}"