#!/usr/bin/env node
import 'source-map-support/register';
import { App, Stack, StackProps, Stage, StageProps } from 'aws-cdk-lib';
import { EC2Stack } from "../lib/ec2Stack";
import { S3Stack } from '../lib/s3Stack';

const DEFAULT_REGION = "us-west-2";
const envSetting = {
  env: {
    account: process.env.CDK_DEPLOY_ACCOUNT || process.env.CDK_DEFAULT_ACCOUNT,
    region: DEFAULT_REGION,
  },
};

const app = new App();

const s3Stack = new S3Stack(app, 'S3Stack', envSetting);

const ec2Stack = new EC2Stack(app, 'EC2Stack', envSetting);
ec2Stack.addDependency(s3Stack); 

// TODO: 기타 스택 차례로 추가

app.synth();
