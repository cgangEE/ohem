#!/usr/bin/python

import google.protobuf.text_format
import google.protobuf as pb
import caffe
import os



def modifyModel(src_model):

    model = caffe.proto.caffe_pb2.NetParameter()

    with open(src_model) as f:
        pb.text_format.Merge(f.read(), model)
         
    dst_model = os.path.splitext(src_model)[0] + 'X.pt'
    upperLayer = False
    eps = 1e-3
        
    for i, layer in enumerate(model.layer):
        if 'conv3' in layer.name:
            upperLayer = True

        for p in layer.param:
            if upperLayer and p.lr_mult > 0.1 - eps \
                    and p.lr_mult < 1 - eps:
                p.lr_mult *= 10
                p.decay_mult *= 10

    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))


if __name__ == '__main__':
    modifyModel('original.pt')
