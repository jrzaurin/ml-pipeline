#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

algorithm_name=ml-pipeline-lightgbm-hyperopt
profile_name=ml_pipelines

chmod +x lightgbm/train
chmod +x lightgbm/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text --profile ${profile_name})

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region --profile ${profile_name})
region=${region:-eu-west-1}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${algorithm_name}" --profile "${profile_name}"> /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" --profile "${profile_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email --profile ${profile_name})

# Build the docker image locally with the algorithm_name and then push it to ECR
# with the full name.

docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}
