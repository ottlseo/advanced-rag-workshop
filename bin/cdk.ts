#!/usr/bin/env node
import "source-map-support/register";
import { App } from "aws-cdk-lib";
import { EC2Stack } from "../lib/ec2Stack";
import { OpensearchStack } from "../lib/openSearchStack";
import { SagemakerNotebookStack } from "../lib/sagemakerNotebookStack";
import * as fs from 'fs';

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

// Deploy EC2 stack
const ec2Stack = new EC2Stack(app, "EC2Stack", envSetting);
ec2Stack.addDependency(opensearchStack);

// Deploy Reranker stack using cloudformation template 
const rerankerStackTemplate = JSON.parse(fs.readFileSync('../cfn/RerankerStack.template.json', 'utf-8'));
app.node.addMetadata('cfn_nag', {
  entries: [
    {
      id: 'RerankerStack',
      data: rerankerStackTemplate,
      info: 'RerankerStack',
      parents: [ec2Stack.stackId]
    }
  ]
});

app.synth();
