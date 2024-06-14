import { Duration, Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";

import * as fs from "fs";
import * as cdk from "aws-cdk-lib";
import * as opensearch from "aws-cdk-lib/aws-opensearchservice";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";

export class OpensearchStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const domainName = `rag-hol-mydomain`;

    const opensearch_user_id = "raguser";
    const opensearch_user_password = "MarsEarth1!";

    const secret = new secretsmanager.Secret(this, "domain-creds", {
      generateSecretString: {
        secretStringTemplate: JSON.stringify({
          "es.net.http.auth.user": opensearch_user_id,
        }),
        generateStringKey: opensearch_user_password,
      },
    });

    const domain = new opensearch.Domain(this, "Domain", {
      version: opensearch.EngineVersion.OPENSEARCH_2_11,
      domainName: domainName,
      capacity: {
        dataNodeInstanceType: "r6g.large.search",
        dataNodes: 1,
      },
      ebs: {
        volumeSize: 100,
        volumeType: ec2.EbsDeviceVolumeType.GP3,
        enabled: true,
      },
      enforceHttps: true,
      nodeToNodeEncryption: true,
      encryptionAtRest: { enabled: true },
      fineGrainedAccessControl: {
        masterUserName: opensearch_user_id,
        masterUserPassword: secret.secretValueFromJson(
          opensearch_user_password
        ),
      },
    });

    // 인덱싱 Lambda 함수
    const indexingLambda = new lambda.Function(this, "indexing", {
      runtime: lambda.Runtime.PYTHON_3_12,
      code: lambda.Code.fromAsset("../lambda/indexing.py"),
      handler: "index.handler",
      environment: {
        OPENSEARCH_DOMAIN: domain.domainEndpoint,
        OPENSEARCH_USER_ID: opensearch_user_id,
        OPENSEARCH_USER_PASSWORD: opensearch_user_password,
      },
    });

    // CloudWatch 이벤트 규칙 생성
    const domainCreatedRule = new events.Rule(this, "DomainCreatedRule", {
      eventPattern: {
        source: ["aws.opensearchservice"],
        detailType: ["OpenSearch Domain Creation"],
        detail: {
          "domainStatus.domainName": [domain.domainName],
          "domainStatus.state": ["Active"],
        },
      },
    });

    // 이벤트 규칙에 Lambda 함수 추가
    domainCreatedRule.addTarget(new targets.LambdaFunction(indexingLambda));
    new cdk.CfnOutput(this, "OpensearchDomainEndpoint", {
      value: domain.domainEndpoint,
      description: "OpenSearch Domain Endpoint",
    });
  }
}
