# Object Detection with AWS: A Demo with Boots and Cats

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AWSRekognitionS3AclBucketRead20191011",
            "Effect": "Allow",
            "Principal": {
                "Service": "rekognition.amazonaws.com"
            },
            "Action": [
                "s3:GetBucketAcl",
                "s3:GetBucketLocation"
            ],
            "Resource": "arn:aws:s3:::gt-object-detect-thewsey-us-east-1"
        },
        {
            "Sid": "AWSRekognitionS3GetBucket20191011",
            "Effect": "Allow",
            "Principal": {
                "Service": "rekognition.amazonaws.com"
            },
            "Action": [
                "s3:GetObject",
                "s3:GetObjectAcl",
                "s3:GetObjectVersion",
                "s3:GetObjectTagging"
            ],
            "Resource": "arn:aws:s3:::gt-object-detect-thewsey-us-east-1/*"
        },
        {
            "Sid": "AWSRekognitionS3ACLBucketWrite20191011",
            "Effect": "Allow",
            "Principal": {
                "Service": "rekognition.amazonaws.com"
            },
            "Action": "s3:GetBucketAcl",
            "Resource": "arn:aws:s3:::gt-object-detect-thewsey-us-east-1"
        },
        {
            "Sid": "AWSRekognitionS3PutObject20191011",
            "Effect": "Allow",
            "Principal": {
                "Service": "rekognition.amazonaws.com"
            },
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::gt-object-detect-thewsey-us-east-1/*",
            "Condition": {
                "StringEquals": {
                    "s3:x-amz-acl": "bucket-owner-full-control"
                }
            }
        }
    ]
}
```
