import { Stack, StackProps, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export class EC2Stack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const defaultVpc = ec2.Vpc.fromLookup(this, 'VPC', {
      isDefault: true,
    })

    const machineImage = ec2.MachineImage.fromSsmParameter(
      '/aws/service/canonical/ubuntu/server/focal/stable/current/amd64/hvm/ebs-gp2/ami-id'
    )

    // EC2 인스턴스 생성
    const webserverInstance = new ec2.Instance(this, 'appInstance', {
      instanceType: new ec2.InstanceType('m5.large'),
      machineImage: machineImage,
      vpc: defaultVpc,
    });

    const webserverSecurityGroup = new ec2.SecurityGroup(this, 'myVmSecurityGroup', {
      vpc: defaultVpc,
    })
    webserverSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'httpIpv4',
    )
    webserverSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv6(),
      ec2.Port.tcp(80),
      'httpIpv6',
    )
  }
}