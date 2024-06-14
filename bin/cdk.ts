#!/usr/bin/env node
import "source-map-support/register";
import { App, Stack, StackProps, Stage, StageProps } from "aws-cdk-lib";
import { EC2Stack } from "../lib/ec2Stack";
import { S3Stack } from "../lib/s3Stack";
import { OpensearchStack } from "../lib/openSearchStack";
import { SagemakerStack, HuggingFaceLlmProps } from "../lib/sagemakerStack";

const DEFAULT_REGION = "us-west-2";
const envSetting = {
  env: {
    account: process.env.CDK_DEPLOY_ACCOUNT || process.env.CDK_DEFAULT_ACCOUNT,
    region: DEFAULT_REGION,
  },
};

const app = new App();

const s3Stack = new S3Stack(app, "S3Stack", envSetting);

const ec2Stack = new EC2Stack(app, "EC2Stack", envSetting);
ec2Stack.addDependency(s3Stack);

const opensearchStack = new OpensearchStack(app, "OpensearchStack", envSetting);
// TODO: 기타 스택 차례로 추가

const HFprops: HuggingFaceLlmProps = {
  name: "reranker",
  instanceType: "ml.g5.xlarge",
  environmentVariables: {
    HF_MODEL_ID: "Dongjin-kr/ko-reranker",
    HF_TASK: "text-classification",
  },
};
const sagemakerStack = new SagemakerStack(app, "SagemakerStack", HFprops);

app.synth();
