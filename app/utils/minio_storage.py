import io
from datetime import timedelta
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from app.core.config import settings

client = Minio(
    endpoint    = settings.MINIO_ENDPOINT,
    access_key  = settings.MINIO_ACCESS_KEY,
    secret_key  = settings.MINIO_SECRET_KEY,
    secure      = settings.MINIO_SECURE,
)


def ensure_bucket(bucket: str = settings.MINIO_BUCKET) -> None:
    if not client.bucket_exists(bucket_name=bucket):
        client.make_bucket(bucket_name=bucket)


def check_connection() -> tuple[bool, str]:
    """Ping MinIO and make sure the default bucket exists. Returns (ok, detail)."""
    try:
        ensure_bucket()
        return True, f"{settings.MINIO_ENDPOINT} (bucket: {settings.MINIO_BUCKET})"
    except Exception as e:
        return False, f"{settings.MINIO_ENDPOINT} - {e}"

def upload_bytes(
    object_name : str,
    data        : bytes,
    content_type: str = "application/octet-stream",
    bucket      : str = settings.MINIO_BUCKET,
) -> str:
    client.put_object(
        bucket_name = bucket,
        object_name = object_name,
        data        = io.BytesIO(data),
        length      = len(data),
        content_type= content_type,
    )
    return object_name


def upload_stream(
    object_name : str,
    stream      : BinaryIO,
    content_type: str = "application/octet-stream",
    bucket      : str = settings.MINIO_BUCKET,
) -> str:
    """Upload a stream of unknown length (e.g. FastAPI UploadFile.file)."""
    client.put_object(
        bucket_name = bucket,
        object_name = object_name,
        data        = stream,
        length      = -1,
        part_size   =10 * 1024 * 1024,
        content_type=content_type,
    )
    return object_name


def upload_file(
    object_name: str,
    file_path: str,
    content_type: str = "application/octet-stream",
    bucket: str = settings.MINIO_BUCKET,
) -> str:
    client.fput_object(
        bucket_name  = bucket,
        object_name  = object_name,
        file_path    = file_path,
        content_type = content_type,
    )
    return object_name


def download_bytes(object_name: str, bucket: str = settings.MINIO_BUCKET) -> bytes:
    response = client.get_object(bucket_name=bucket, object_name=object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def object_exists(object_name: str, bucket: str = settings.MINIO_BUCKET) -> bool:
    try:
        client.stat_object(bucket_name=bucket, object_name=object_name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        raise

def presigned_url(
    object_name : str,
    expires     : timedelta = timedelta(hours=1),
    bucket      : str = settings.MINIO_BUCKET,
) -> str:
    return client.presigned_get_object(
        bucket_name=bucket, object_name=object_name, expires=expires
    )


def delete_object(object_name: str, bucket: str = settings.MINIO_BUCKET) -> None:
    client.remove_object(bucket_name=bucket, object_name=object_name)