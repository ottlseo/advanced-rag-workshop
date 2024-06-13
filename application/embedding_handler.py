import os, boto3
import uuid

s3 = boto3.client('s3')

CUSTOM_FILE_BUCKET_NAME = "adv-rag-custom-docs-bucket"

def generate_random_string(length):
    random_str = str(uuid.uuid4())
    random_str = random_str.replace("-", "")  # '-' 문자를 제거
    return random_str[:length]

def upload_file_to_s3(bucket_name, file):
    key_name = generate_random_string(8) + "-" + os.path.basename(file.name)
    try:
        s3.upload_fileobj(file, bucket_name, key_name) # 파일을 S3에 업로드
        return key_name
    except Exception as e:
        return e

def upload_file_to_custom_docs_bucket(file):
    return upload_file_to_s3(CUSTOM_FILE_BUCKET_NAME, file)