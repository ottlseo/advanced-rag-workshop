import os
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection

# OpenSearch 도메인 엔드포인트
DOMAIN_ENDPOINT = os.environ['OPENSEARCH_DOMAIN']

# OpenSearch 사용자 인증 정보
OPENSEARCH_USER_ID = os.environ['OPENSEARCH_USER_ID']
OPENSEARCH_USER_PASSWORD = os.environ['OPENSEARCH_USER_PASSWORD']

# OpenSearch 클라이언트 초기화
def get_opensearch_client():
    client = OpenSearch(
        hosts=[{'host': DOMAIN_ENDPOINT, 'port': 443}],
        http_auth=(OPENSEARCH_USER_ID, OPENSEARCH_USER_PASSWORD),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    return client

# Lambda 핸들러 함수
def handler(event, context):
    # OpenSearch 클라이언트 인스턴스 생성
    client = get_opensearch_client()

    # 인덱싱할 데이터 (예: S3 이벤트에서 받은 데이터)
    data = event['Records'][0]['s3']['object']['key']

    # 인덱싱 작업 수행
    response = client.index(
        index='your-index-name',
        body=json.dumps({'data': data}),
        refresh=True
    )

    print(f'Indexing response: {response}')

    return {
        'statusCode