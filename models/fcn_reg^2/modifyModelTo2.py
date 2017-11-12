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

        for j, top in enumerate(layer.top):
            layer.top[j] += '/twice'

        for j, bottom in enumerate(layer.bottom):
            layer.bottom[j] += '/twice'

        layer.name += '/twice'


    dst_model = os.path.splitext(src_model)[0] + '2.pt'
    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))


if __name__ == '__main__':
    modifyModel('original.pt')
