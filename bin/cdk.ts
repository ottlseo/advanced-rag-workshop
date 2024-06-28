#!/usr/bin/env node
import "source-map-support/register";
import { App } from "aws-cdk-lib";
import { EC2Stack } from "../lib/ec2Stack/ec2Stack";
import { OpensearchStack } from "../lib/openSearchStack";
import { SagemakerNotebookStack } from "../lib/sagemakerNotebookStack/sagemakerNotebookStack";
import { CfnInclude } from 'aws-cdk-lib/cloudformation-include';

const DEFAULT_REGION = "us-west-2";
const envSetting = {
  env: {
    account: process.env.CDK_DEPLOY_ACCOUNT || process.env.CDK_DEFAULT_ACCOUNT,
    region: DEFAULT_REGION,
  },
};

const app = new App();

// Deploy Sagemaker stack
const sagemakerNotebookStack = new SagemakerNotebookStack(app, "SagemakerNotebookStack", envSetting);

// Deploy OpenSearch stack
const opensearchStack = new OpensearchStack(app, "OpensearchStack", envSetting);
opensearchStack.addDependency(sagemakerNotebookStack);

// Deploy Reranker stack using cloudformation template 
const rerankerStack = new CfnInclude(opensearchStack, 'RerankerStack', {
  templateFile: 'lib/rerankerStack/RerankerStack.template.json'
});

// Deploy EC2 stack
const ec2Stack = new EC2Stack(app, "EC2Stack", envSetting);
ec2Stack.addDependency(opensearchStack);
ec2Stack.node.addDependency(rerankerStack);

app.synth();
