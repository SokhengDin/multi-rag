from pydantic_settings import BaseSettings
from decouple import config

class Settings(BaseSettings):

    API_PORT: int   = config('API_PORT', cast=int)

    DB_HOST: str = config('DB_HOST', cast=str)
    DB_PORT: str = config('DB_PORT', cast=str)
    DB_NAME: str = config('DB_NAME', cast=str)
    DB_USER: str = config('DB_USER', cast=str)
    DB_PASS: str = config('DB_PASS', cast=str)

    MINIO_ACCESS_KEY: str = config('MINIO_ACCESS_KEY', cast=str)
    MINIO_SECRET_KEY: str = config('MINIO_SECRET_KEY', cast=str)
    MINIO_ENDPOINT  : str = config('MINIO_ENDPOINT', cast=str)
    MINIO_SECURE    : bool = config('MINIO_SECURE', cast=bool, default=False)
    MINIO_BUCKET    : str = config('MINIO_BUCKET', cast=str, default='documents')

settings = Settings()