#!/usr/bin/env node
import "source-map-support/register";
import { App } from "aws-cdk-lib";
import { EC2Stack } from "../lib/ec2Stack";
import { OpensearchStack } from "../lib/openSearchStack";
import { SageMakerNotebookInstanceStack } from "../lib/sagemakerNotebookStack";

const DEFAULT_REGION = "us-west-2";
const envSetting = {
  env: {
    account: process.env.CDK_DEPLOY_ACCOUNT || process.env.CDK_DEFAULT_ACCOUNT,
    region: DEFAULT_REGION,
  },
};

const app = new App();

const opensearchStack = new OpensearchStack(app, "OpensearchStack", envSetting);

const ec2Stack = new EC2Stack(app, "EC2Stack", envSetting);
ec2Stack.addDependency(opensearchStack);

new SageMakerNotebookInstanceStack(app, "SagemakerNotebookInstance", envSetting);

app.synth();
