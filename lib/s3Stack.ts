import { Stack, StackProps, aws_s3 as s3, CfnOutput, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as cdk from 'aws-cdk-lib';

export class S3Stack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const accountId = Stack.of(this).account;

    // S3 버킷 생성
    const bucketName = `adv-rag-custom-docs-bucket-${accountId}`;
    const bucket = new s3.Bucket(this, 'CustomBucket', {
      bucketName: bucketName,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: true,
    });

    new CfnOutput(this, 'BucketName', {
      value: bucket.bucketName,
      description: 'S3 bucket name for custom file upload',
    });
  }
}
