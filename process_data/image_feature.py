import os
import glob
import re
import PIL
import h5py
import torch as t
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from process_data.convnet import factory
from .data_preprocess import get_games
from .image_process import get_transform, crop_object
default_collate = t.utils.data.dataloader.default_collate

transform = get_transform((224, 224))

class ImageSet(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        filename = self.filenames[item]
        try:
            raw_img = PIL.Image.open(filename).convert('RGB')
            img = transform(raw_img)  # 3*224*224 Tensor
        except:
            return None
        id = int(re.match(r'.*_0*(\d+)\.\w+$', filename).group(1))
        return img, id


def collate(batch):
    batch = [x for x in batch if x is not None]
    batch = default_collate(batch)
    return batch


class ObjectSet(Dataset):
    def __init__(self, games, image_dir):
        self.games = games
        self.image_dir = image_dir

    def __len__(self):
        return len(self.games)

    def __getitem__(self, item):
        game = self.games[item]
        img = game.img
        try:
            raw_img = PIL.Image.open(os.path.join(self.image_dir, img.filename)).convert('RGB')
        except:
            return None
        object = game.object
        crop = crop_object(object.bbox, raw_img)
        crop = transform(crop)
        return crop, object.id


def extract_image_feature(flags):
    # 抽取图像vgg16  fc8的特征
    is_cuda = t.cuda.is_available()  # 是否有GPU资源
    convnet = factory(flags, is_cuda, False)

    names = glob.glob(os.path.join(flags.image_dir, "COCO_*"))
    imageset = ImageSet(names)
    batch_size = flags.batch_size
    imageloader = DataLoader(imageset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    size = len(names)
    print("total images: {}".format(size))
    print("total batchs: {}".format(size/batch_size))

    shape = tuple([size] + list(flags.shape))
    fea_dir = os.path.join(flags.fea_dir, flags.arch)
    if not os.path.exists(fea_dir):
        os.mkdir(fea_dir)
        print("make dir")
    fea_dir = os.path.join(fea_dir, flags.feature)
    f = h5py.File(os.path.join(fea_dir, "image.hdf5"), "w")
    fea_set = f.create_dataset("feature", shape, chunks=True)
    index_set = f.create_dataset("index", (size,), dtype='i')

    index = 0
    for batch_input in tqdm(imageloader):
        # retrieve id images
        fea, ids = batch_input
        fea = convnet(fea).detach().cpu().numpy()
        size = ids.shape[0]
        fea_set[index:index+size] = fea
        index_set[index:index+size] = ids
        index += size

    f.close()
    print("extract feature finish")


def get_image_feature(feature_dir, id):
    # 根据image id 获取vgg16 fc8特征
    f = h5py.File(os.path.join(feature_dir, "image.hdf5"), "r")
    fea_set = f["feature"]
    key2id = f["index"]
    print(key2id[:10])
    id2key = {v:k for k,v in enumerate(key2id)}
    key = id2key[id]
    fea = fea_set[key]
    print(fea)
    return fea


def extract_object_feature(flags):
    is_cuda = t.cuda.is_available()  # 是否有GPU资源
    cnn = factory(flags, is_cuda, is_cuda)

    games = []
    for set in ["train", "valid", "test"]:
        games.extend(get_games(flags.data_dir, set))

    objectset = ObjectSet(games, flags.image_dir)
    batch_size = flags.batch_size
    objectloader = DataLoader(objectset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    fea_dir = os.path.join(flags.fea_dir, flags.arch)
    if not os.path.exists(fea_dir):
        os.mkdir(fea_dir)
    fea_dir = os.path.join(fea_dir, flags.feature)
    f = h5py.File(os.path.join(fea_dir, "crop.hdf5"), "w")
    shape = tuple([len(objectset)] + list(flags.shape))
    fea_set = f.create_dataset("feature", shape, chunks=True)
    index_set = f.create_dataset("index", (len(objectset),), dtype='i')

    index = 0
    for batch_input in tqdm.tqdm(objectloader):
        # retrieve id images
        fea, ids = batch_input
        fea = cnn(fea).detach().cpu().numpy()
        size = ids.shape[0]
        fea_set[index:index + size] = fea
        index_set[index:index + size] = ids
        index += size

    f.close()


if __name__ ==  "__main__":
    # extract_image_feature()
    # extract_object_feature()
    # get_image_feature(106073)
    pass

