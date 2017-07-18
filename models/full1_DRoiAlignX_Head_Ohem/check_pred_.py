#!/usr/bin/python

import google.protobuf.text_format
import google.protobuf as pb
import caffe
import os



def modifyModel(src_model):

    model = caffe.proto.caffe_pb2.NetParameter()

    with open(src_model) as f:
        pb.text_format.Merge(f.read(), model)
         
        
    for i, layer in enumerate(model.layer):
        if 'pred_' in layer.name:
            print(layer)


if __name__ == '__main__':
    modifyModel('original.pt')
