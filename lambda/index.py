import json
import boto3
import logging
import os
import cfnresponse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

opensearch = boto3.client('opensearch')

def lambda_handler(event, context):
    print(event)
    #domain_name = f"chatbot-{os.environ['AWS_STACK_NAME']}"
    props = event["ResourceProperties"]
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
    
    try:
        if event['RequestType'] == 'Delete':
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
        elif event['RequestType'] == 'Create' or event['RequestType'] == 'Update':
            response = opensearch.associate_package(
                PackageID=package_id,  # Nori plugin Package ID for us-west-2 and version 2.11
                DomainName=domain_name
            )
            filtered_response = {
                key: value for key, value in response.items() if key in ['Status', 'PackageID']
            }
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {'Data': filtered_response})
    except Exception as e:
        logger.error(f"Failed to associate package: {e}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {'Message': str(e)})
