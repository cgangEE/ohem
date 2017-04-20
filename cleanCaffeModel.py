#!/usr/bin/python

import os

def getCaffeIterator(model):
    ret = 0
    for c in model:
        if c>='0' and c<='9':
            ret = ret * 10 + int(c)
        elif ret != 0:
            return ret
    return ret



def cleanCaffeModel(dirname):
    for parent, dirnames, filenames in os.walk(dirname):
        modelList = []

        for filename in filenames:
            if filename.find('.caffemodel') != -1:
                modelList.append(filename);

        if len(modelList) > 0:
            print(parent)

            modelList.sort(key = getCaffeIterator)
            for model in modelList[:-1]:
                os.remove(os.path.join(parent, model))

if __name__ == '__main__':
    cleanCaffeModel('output')
