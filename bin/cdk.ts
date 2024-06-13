#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { App, Stack, StackProps, Stage, StageProps } from 'aws-cdk-lib';
import { EC2Stack } from "../lib/ec2Stack";

const DEFAULT_REGION = "us-west-2";

const app = new App();

// Stage 생성
const ec2Stage = new Stage(app, 'EC2Stage', {
  env: {
    account: process.env.CDK_DEPLOY_ACCOUNT || process.env.CDK_DEFAULT_ACCOUNT,
    region: DEFAULT_REGION,
  },
});

// EC2Stack 생성 및 Stage에 추가
new EC2Stack(ec2Stage, 'EC2Stack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: DEFAULT_REGION,
  },
});

// TODO: 기타 스택 차례로 추가

app.synth();
