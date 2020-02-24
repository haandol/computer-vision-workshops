# Built-Ins:
import os
import sys
import json
import time
import logging
import argparse
import warnings
import subprocess
from base64 import b64decode

# Install/Update GluonCV:
subprocess.call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'gluoncv'])

import mxnet as mx
import gluoncv as gcv


def model_fn(model_dir):
    ctx = mx.cpu()
    classes = ['person']
    net = gcv.model_zoo.get_model(
        'yolo3_darknet53_custom',
        classes=classes,
        pretrained_base=False,
        ctx=ctx,
    )
    net.load_parameters(os.path.join(model_dir, 'model.params'), ctx=ctx)
    return net


def input_fn(request_body, content_type):
    if content_type == 'application/json':
        D = json.loads(request_body)
        
        short = D.get('width', 416)
        image = b64decode(D['image'])
        x, _ = gcv.data.transforms.presets.yolo.transform_test(
            mx.image.imdecode(image), short=short
        )
        return x
    else:
        raise RuntimeError(f'Not support content-type: {content_type}')


def predict_fn(input_object, model):
    cid, score, bbox = model(input_object)
    return cid, score, bbox


def output_fn(prediction, content_type):
    cid, score, bbox = prediction
    if content_type == 'application/json':
        return json.dumps({
            'cid': cid.asnumpy().tolist(),
            'score': score.asnumpy().tolist(),
            'bbox': bbox.asnumpy().tolist()
        })
    else:
        raise RuntimeError(f'Not support content-type: {content_type}')