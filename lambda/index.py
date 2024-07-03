import json
import boto3
import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

opensearch = boto3.client('opensearch')
secrets_manager = boto3.client('secretsmanager')

def on_event(event, context):
    print(event)
    request_type = event['RequestType']
    if request_type == 'Create': return on_create(event)
    if request_type == 'Update': return on_update(event)
    if request_type == 'Delete': return on_delete(event)
    raise Exception("Invalid request type: %s" % request_type)

def on_create(event):
    props = event["ResourceProperties"]
    print("create new resource with props %s" % props)

    domain_arn = props['DomainArn']
    domain_name = domain_arn.split('/')[-1]
    DEFAULT_REGION = props['DEFAULT_REGION']
    VERSION = props['VERSION']
    
    nori_pkg_id = {}
    nori_pkg_id['us-east-1'] = {
        '2.3': 'G196105221',
        '2.5': 'G240285063',
        '2.7': 'G16029449', 
        '2.9': 'G60209291',
        '2.11': 'G181660338'
    }
    
    nori_pkg_id['us-west-2'] = {
        '2.3': 'G94047474',
        '2.5': 'G138227316',
        '2.7': 'G182407158', 
        '2.9': 'G226587000',
        '2.11': 'G79602591'
    }

    #package_id = "G79602591" #props['PackageID']
    package_id = nori_pkg_id[DEFAULT_REGION][VERSION] 
    print(domain_arn, domain_name, package_id)
        
    #opensearch_domain_endpoint = event['ResourceProperties']['OpensearchDomainEndpoint']
        
    #response = secrets_manager.get_secret_value(
    #    SecretId='opensearch_user_password'
    #)
    #secrets_string = response.get('SecretString')
    #secrets_dict = eval(secrets_string)
    #opensearch_user_id = secrets_dict['es.net.http.auth.user']
    #opensearch_user_password = secrets_dict['pwkey']
        

    # Associate the "nori" package with the OpenSearch Domain
    #response = opensearch.associate_package(
    #    PackageID=package_id,
    #    DomainName=domain_name
    #)
    
    # Retry logic
    max_retries = 10
    retries = 0

    while retries < max_retries:
        try:
            response = opensearch.associate_package(
                PackageID=package_id,
                DomainName="rag-hol-test" #domain_name
            )
            break
        except opensearch.exceptions.BaseException as e:
            if "A concurrent operation (associate/dissociate) is in progress for this domain" in str(e):
                retries += 1
                time.sleep(30)  # Wait 30 seconds before retrying
            else:
                raise e
    
    # Return the physical resource ID
    physical_id = f"AssociatePackage-{domain_name}-{package_id}"
    return { 'PhysicalResourceId': physical_id }

def on_update(event):
    physical_id = event["PhysicalResourceId"]
    props = event["ResourceProperties"]
    print("update resource %s with props %s" % (physical_id, props))
    # Update logic if needed
    return { 'PhysicalResourceId': physical_id }

def on_delete(event):
    physical_id = event["PhysicalResourceId"]
    print("delete resource %s" % physical_id)
    # Delete logic if needed
"""
def is_complete(event, context):
    physical_id = event["PhysicalResourceId"]
    request_type = event["RequestType"]
    props = event["ResourceProperties"]

    domain_arn = props['DomainArn']
    domain_name = domain_arn.split('/')[-1]
    package_id = "G79602591" #props['PackageID']

    # Check the association status
    response = opensearch.describe_package(
        PackageID=package_id
    )

    # Extract the domain associations from the response
    domain_associations = response.get('DomainPackageDetailsList', [])

    # Determine if the package is successfully associated
    is_ready = any(
        association['DomainName'] == domain_name and 
        association['DomainPackageStatus'] == 'ACTIVE'
        for association in domain_associations
    )

    return { 'IsComplete': is_ready }
"""