# mrec-mlflow-examples
This is a collection of DVC project examples that you can directly run with mlflow CLI commands or directly using
 Python.

The goal is provide you with additional set of samples, focusing on machine learning and deep learning examples, to
 get you quickly started on MlFlow.



## How to get set up on Amazon
```
# Setting up EC2 Server
$ sudo yum install git python3 -y
$ sudo pip3 install mlflow
$ sudo pip3 install psycopg2-binary boto3
## Export AWS Secret Key
$ export AWS_ACCESS_KEY_ID= <ACCESS_KEY>
$ export AWS_SECRET_ACCESS_KEY= <SECRET_KEY>

# Set up S3 bucket for artifact store
simply set up the s3 bucket and get the S3 URL
--default-artifact-root = s3://mrec-s3-bucket/medical_relations/mlflow

# Set up model registery
simply set up aws RDS and set up the backend-store-uri
--backend-store-uri = postgresql://USRNAME:PW:<RDS END POINT>:PORT/DATABASE_NAME

# Run server
$ mlflow server --backend-store-uri postgresql://ktl014:XBKEP2F90ImMyTvPitC4@mlflow-rds.cxuxujsmh4fe.us-west-1.rds.amazonaws.com:5432/postgres --default-artifact-root s3://mrec-s3-bucket/medical_relations/mlflow --host 0.0.0.0

# On Client side
export MLFLOW_TRACKING_URI=http://ec2-184-169-233-22.us-west-1.compute.amazonaws.com:5000/
python mrec/train_mrec.py
```
