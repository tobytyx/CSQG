# -*- coding: utf-8 -*-
import gzip
import json
import pickle
from process_data.image_process import get_spatial_feat
from utils.constants import categories, c_len
import copy


class Game(object):
    questions2category = None

    def __new__(cls, *args, **kwargs):
        if not cls.questions2category:
            with open("../data/q_category_single.pkl", mode="rb") as f:
                q_category = pickle.load(f)
            cls.questions2category = q_category["questions2category"]
        return super().__new__(cls)

    def __init__(self, id, status, image, qas, object_id, objects, guess_id=None, **kwargs):
        self.id = id
        self.status = status
        self.img = Image(id=image['id'], width=image['width'], height=image['height'], filename=image['file_name'])
        self.object_id = object_id
        self.guess_id = guess_id
        self.questions = [q['question'] for q in qas]
        self.answers = [q['answer'] for q in qas]
        self.q_cates = []  # record category
        self.gt_q_cates = []  # record gt category for loop
        self.question_ids = [q['id'] for q in qas]
        self.objects = []
        for q in self.questions:
            q_cate = [0] * c_len
            for cate in self.questions2category[q]:
                q_cate[categories[cate]] = 1
            self.q_cates.append(q_cate)
            self.gt_q_cates.append(copy.copy(q_cate))

        for o in objects:
            obj = Object(id=o['id'],
                         category=o['category'],
                         category_id=o['category_id'],
                         bbox=o['bbox'],
                         segment=o['segment'],
                         image=self.img
                         )
            self.objects.append(obj)
            if o['id'] == self.object_id:
                self.object = obj

    def __str__(self):
        qas = []
        for q, a in zip(self.questions, self.answers):
            qas.append(q+" " + a + "\n")
        if len(self.questions) > len(self.answers):  # when question more than answers, print the question
            qas.append(self.questions[-1])
        return "game id: {0}\nqas:\n{1}object_id: {2}\nstatus: {3}".\
            format(self.id, " ".join(qas), self.object.id, self.status)


class GameMultiCate(Game):
    def __new__(cls, *args, **kwargs):
        if not cls.questions2category:
            print("multi categories.")
            with open("../data/q_category_multi.pkl", mode="rb") as f:
                q_category = pickle.load(f)
            cls.questions2category = q_category["questions2category"]
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, id, status, image, qas, object_id, objects, guess_id=None, **kwargs):
        super(GameMultiCate, self).__init__(id, status, image, qas, object_id, objects, guess_id=None, **kwargs)


class Image(object):
    def __init__(self, id, width, height, filename):
        self.id = id
        self.width = width
        self.height = height
        self.filename = filename

    def __str__(self):
        return str({"file_name": self.filename, "height": self.height, "width": self.width, "id": self.id})


class Object(object):
    def __init__(self, id, category, category_id, segment, bbox, image):
        self.id = id
        self.category = category
        self.category_id = category_id
        self.segment = segment
        self.bbox = BBox(bbox)
        self.spatial = get_spatial_feat(self.bbox, image.width, image.height)

    def __str__(self):
        return str({"id": self.id, "category": self.category, "category_id": self.category_id,
                    "segment": self.segment, "bbox": eval(self.bbox.__str__())})

    def json(self):
        return {
            "id": self.id, "category": self.category, "category_id": self.category_id,
            "segment": self.segment, "bbox": self.bbox.json()
        }


class BBox(object):
    def __init__(self, bbox):
        # bbox 	[x,y,width,height]
        self.width = bbox[2]
        self.height = bbox[3]
        self.x = bbox[0]
        self.y = bbox[1]

    def __str__(self):
        return str([self.x, self.y, self.width, self.height])

    def json(self):
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def get_games(data_dir, dataset, multi_cate=False):
    games = []
    file = '{}/guesswhat.{}.jsonl.gz'.format(data_dir, dataset)
    with gzip.open(file) as f:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            if multi_cate:
                g = GameMultiCate(**game)
            else:
                g = Game(**game)
            games.append(g)
    return games


if __name__ == "__main__":
    # 一轮game样例数据
    data = {"status": "success", "qas": [{"answer": "No", "question": "is it in the sky?", "id": 5200},
                                         {"answer": "No", "question": "is it the umbrella?", "id": 5208},
                                         {"answer": "No", "question": "is it the ocean?", "id": 5219},
                                         {"answer": "Yes", "question": "is it the lifeboat?", "id": 5223}],
            "questioner_id": 34, "timestamp": "2016-07-08 15:13:03",
            "image": {"file_name": "COCO_train2014_000000175527.jpg", "coco_url": "http://mscoco.org/images/175527",
                      "height": 426, "width": 640,
                      "flickr_url": "http://farm4.staticflickr.com/3281/2552062880_d776f7fdb7_z.jpg", "id": 175527},
            "object_id": 179938, "objects": [
            {"category": "umbrella", "area": 5487.90535, "bbox": [398.57, 286.02, 136.26, 87.92], "category_id": 28,
             "segment": [
                 [482.74, 298.14, 499.41, 308.15, 516.91, 326.48, 534.83, 350.65, 531.91, 352.31, 528.58, 360.6, 512.74,
                  370.19, 508.99, 373.94, 501.91, 370.19, 498.99, 363.52, 486.49, 359.77, 482.74, 349.35, 480.66,
                  344.77, 471.91, 344.77, 463.58, 346.85, 466.91, 343.1, 453.99, 338.52, 445.66, 334.35, 437.32, 325.19,
                  422.74, 322.27, 408.57, 316.85, 409.82, 312.27, 398.57, 304.35, 401.49, 293.93, 408.99, 288.1, 420.24,
                  286.02]], "id": 1424479},
            {"category": "kite", "area": 1891.0183, "bbox": [437.49, 86.82, 45.95, 81.37], "category_id": 38,
             "segment": [
                 [437.49, 101.18, 465.25, 86.82, 478.65, 120.33, 482.48, 128.94, 483.44, 157.66, 471.95, 168.19, 444.19,
                  113.63]], "id": 622400},
            {"category": "boat", "area": 1687.0649500000013, "bbox": [536.47, 374.05, 78.15, 36.21], "category_id": 9,
             "segment": [
                 [536.47, 388.34, 546.0, 379.77, 549.81, 379.77, 549.81, 374.05, 559.35, 375.0, 562.2, 379.77, 589.84,
                  385.48, 598.42, 383.58, 600.32, 375.95, 604.14, 377.86, 602.23, 383.58, 610.81, 391.2, 614.62, 395.01,
                  609.86, 407.4, 606.04, 409.31, 565.06, 410.26, 563.16, 386.44, 559.35, 384.53, 551.72, 387.39, 553.63,
                  407.4, 542.19, 403.59, 537.43, 389.3]], "id": 179938},
            {"category": "boat", "area": 206.18160000000017, "bbox": [317.08, 324.85, 28.71, 11.52], "category_id": 9,
             "segment": [
                 [317.08, 330.36, 321.09, 334.87, 330.1, 336.37, 345.79, 335.87, 345.29, 333.53, 342.62, 331.36, 342.62,
                  331.36, 331.43, 328.69, 329.26, 325.19, 319.42, 324.85, 319.25, 327.86]], "id": 180748}], "id": 2488}

    for item in data.items():
        print(item)
    game = Game(**data)
    print(game.img)
    print(eval(game.img.__str__()))
    print(type(eval(game.img.__str__())))
    print(eval(game.object.__str__()))

    # get_games(data_dir = "./../../data/", set='train')