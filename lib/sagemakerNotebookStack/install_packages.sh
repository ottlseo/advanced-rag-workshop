#!/bin/bash

set -e


sudo -u ec2-user -i <<'EOF'
 
#source /home/ec2-user/anaconda3/bin/deactivate
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U pip
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U awscli
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U botocore
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U boto3
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U sagemaker 
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U langchain
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U langchain-community
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U langchain_aws
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U termcolor
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U transformers
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U librosa
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U opensearch-py
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U sqlalchemy #==2.0.1
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U pypdf
/home/ec2-user/anaconda3/envs/python3/bin/python -m #pip install -U spacy
# /home/ec2-user/anaconda3/envs/python3/bin/python -m spacy download ko_core_news_md
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U ipython
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U ipywidgets
#/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U llmsherpa
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U anthropic
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U faiss-cpu
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U jq
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U pydantic

sudo rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo yum -y update
sudo yum install -y poppler-utils
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U lxml
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U kaleido
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U uvicorn
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U pandas
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U numexpr
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U pdf2image

#sudo sh install_tesseract.sh
#sudo sh SageMaker/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/00_setup/install_tesseract.sh
sudo amazon-linux-extras install libreoffice -y
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U "unstructured[all-docs]"
#sudo rm -rf leptonica-1.84.1 leptonica-1.84.1.tar.gz tesseract-ocr

/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U python-dotenv
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U llama-parse
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U pymupdf

EOF