import os
import PIL
from process_data.convnet import factory
from .image_process import crop_object


class Redis_image_builder(object):
    def __init__(self, redis, id2key, att="id"):
        self.redis = redis
        self.id2key = id2key
        self.att = att

    def load(self, image):
        id = getattr(image, self.att)
        key = self.id2key[id]
        image = self.redis.get(key)
        return image


class H5py_image_builder(object):
    def __init__(self, hfile, id2key, att="id"):
        self.hfile = hfile
        self.id2key = id2key
        self.att = att

    def load(self, image):
        pid = getattr(image, self.att)
        key = self.id2key[pid]
        image = self.hfile[key]
        return image


class Raw_image_builder(object):
    def __init__(self, img_dir, transform, opt, is_cuda=False):
        self.img_dir = img_dir
        self.transform = transform
        self.convnet = factory(opt, is_cuda, is_cuda)

    def load(self, image):
        filename = image.filename
        filename = os.path.join(self.img_dir, filename)
        raw_img = PIL.Image.open(filename).convert('RGB')
        img = self.transform(raw_img)  # 3*224*224 Tensor
        if img.dim() == 3:
            # no batch, only one image
            img = img.unsqueeze(0)
        fea = self.convnet(img).squeeze(0)
        return fea


class Redis_crop_builder(object):
    def __init__(self, redis, id2key):
        self.redis = redis
        self.id2key = id2key

    def load(self, image, object):
        id = object.id
        key = self.id2key[id]
        image = self.redis.get(key)
        return image


class H5py_crop_builder(object):
    def __init__(self, hfile, id2key):
        self.hfile = hfile
        self.id2key = id2key

    def load(self, image, object):
        id = object.id
        key = self.id2key[id]
        image = self.hfile[key]
        return image


class Raw_crop_builder(object):
    def __init__(self, img_dir, transform, opt, is_cuda=False):
        self.img_dir = img_dir
        self.transform = transform
        self.convnet = factory(opt, is_cuda, is_cuda)

    def load(self, image, object):
        filename = image.filename
        filename = os.path.join(self.img_dir, filename)
        raw_img = PIL.Image.open(filename).convert('RGB')
        crop = crop_object(object.bbox, raw_img)
        crop = self.transform(crop)
        if crop.dim() == 3:
            # no batch, only one image
            crop.unsqueeze_(0)
        fea = self.convnet(crop).squeeze_(0)
        return fea


class Redis_object_builder(object):
    def __init__(self, redis, name2i):
        self.redis = redis
        self.name2i = name2i

    def load(self, image):
        name = image.filename
        index = self.name2i[name]
        object = self.redis.get(index)
        return object


class H5py_object_builder(object):
    def __init__(self, hfile, name2i):
        self.hfile = hfile
        self.name2i = name2i

    def load(self, image):
        name = image.filename
        index = self.name2i[name]
        object = self.hfile[index]
        return object
