#!/usr/bin/env node
import "source-map-support/register";
import { App } from "aws-cdk-lib";
import { ApplicationStack } from "../lib/applicationStack";
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

const opensearchStack = new OpensearchStack(app, "OpensearchStack", envSetting);

const HFprops: HuggingFaceLlmProps = {
  name: "reranker",
  instanceType: "ml.g5.xlarge",
  environmentVariables: {
    HF_MODEL_ID: "Dongjin-kr/ko-reranker",
    HF_TASK: "text-classification",
  },
};

const sagemakerStack = new SagemakerStack(app, "SagemakerStack", HFprops);
sagemakerStack.addDependency(opensearchStack);

const applicationStack = new ApplicationStack(app, "ApplicationStack", envSetting);
applicationStack.addDependency(sagemakerStack);

app.synth();
