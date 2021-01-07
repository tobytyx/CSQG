# -*- coding: utf-8 -*-
import os
import json

PAD = 0
SOS = 1
EOS = 2
EOD = 3
UNK = 4
YES = 5
NO = 6
NA = 7

PAD_WORD = '<padding>'
SOS_WORD = '<start>'
EOD_WORD = '<stop_dialogue>'
EOS_WORD = '<stop>'
UNK_WORD = '<unk>'
YES_WORD = '<yes>'
NO_WORD = '<no>'
NA_WORD = '<n/a>'

max_generate_length = 10

BERT_PAD_WORD = '[PAD]'
BERT_UNK_WORD = '[UNK]'
BERT_CLS_WORD = '[CLS]'
BERT_SEP_WORD = '[SEP]'

BERT_PAD = 0
BERT_PAD_MASK = 0
BERT_INPUT_MASK = 1
BERT_PAD_SEG_ID = 0

question_category = {
"q_object": ['person', 'vehicle', 'outdoor', 'animal', 'accessory', 'sports', 'kitchen', 'food', 'furniture',
             'human', 'cloth', 'cloths', 'clothing', 'people','persons', 'electronic', 'appliance', 'indoor',
             'utensil',  'logos', 'logo',
             "gingerbread", "earring", "hinge",
             'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'lettuce', 'ducks', 'duck',
             'truck', 'boat', 'traffic', 'light', 'fire hydrant', 'stop sign', 'parking meter',
             'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
             'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
             'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
             'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'table',
             'toilet', 'toilets', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
             'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush', 'meter', 'bear', 'cell', 'phone',
             'wine', 'glass', 'racket', 'baseball', 'glove', 'hydrant', 'drier', 'kite',
             'sofa', 'fork', 'adult', 'arms', 'baby', 'bag', 'ball', 'bananas', 'basket', 'bat',
             'batter', 'bike', 'birds', 'board', 'body', 'books', 'bottles', 'box', 'boy', 'bread',
             'brush', 'building', 'bunch', 'cabinet', 'camera', 'candle', 'cap', 'carrots', 'cars',
             'cart', 'case', 'catcher', 'cell phone', 'chairs', 'child', 'chocolate', 'coat', 'coffee',
             'computer', 'controller', 'counter', 'cows', 'cupboard', 'cups', 'curtain', 'cycle', 'desk',
             'device', 'dish', 'doll', 'door', 'dress', 'driver', 'equipment', 'eyes',
             'fan', 'feet', 'female', 'fence', 'fire', 'flag', 'flower', 'flowers', 'foot', 'frame',
             'fridge', 'fruit', 'girl', 'girls', 'glasses', 'guy', 'guys', 'hair drier', 'handle',
             'hands', 'hat', 'helmet', 'house', 'jacket', 'jar', 'jeans', 'kid', 'kids', 'lady',
             'lamp', 'leg', 'legs', 'luggage', 'machine', 'male', 'man', 'meat', 'men', 'mirror', 'mobile',
             'monitor', 'mouth', 'mug', 'napkin', 'pan', 'pants', 'paper', 'pen', 'picture', 'pillow',
             'plant', 'plate', 'player', 'players', 'pole', 'pot', 'purse', 'rack', 'racket', 'road',
             'roof', 'screen', 'shelf', 'shelves', 'shirt', 'shoe', 'shoes', 'short', 'shorts', 'shoulder',
             'signal', 'sign', 'silverware', 'skate', 'ski', 'sky', 'snow', 'soap', 'speaker', 'stairs',
             'statue', 'stick', 'stool', 'stove', 'street', 'suit', 'sunglasses', 'suv', 'teddy', 'tennis',
             'tent', 'tomato', 'towel', 'tower', 'toy', 'traffic', 'tray', 'tree', 'trees', 't-shirt', 'ships',
             'tshirt', 'vegetable', 'vest', 'wall', 'watch', 'wheel', 'wheels', 'window', 'windows', 'woman', 'women'
            ],
# ATTRIBUTES
"color": ['white', 'red', 'black', 'blue', 'green', 'yellow', 'orange', 'brown', 'pink',
          'grey', 'gray', 'dark', 'purple', 'color', 'colored', 'colour', 'blond', 'beige', 'bright'],
"location": ['1st', '2nd', 'third', '3', '3rd', 'four', '4th', 'fourth', '5', '5th',
             'five', 'first', 'second', 'last', 'above', 'across', 'after', 'around', 'at',
             'away', 'back', 'background', 'before', 'behind', 'below', 'beside', 'between',
             'bottom', 'center', 'close', 'closer', 'closest', 'corner', 'directly', 'down',
             'edge', 'end', 'entire', 'facing', 'far', 'farthest', 'floor', 'foreground', 'from',
             'front', 'furthest', 'ground', 'hidden', 'in', 'inside', 'left', 'leftmost',
             'middle', 'near', 'nearest', 'next', 'next to', 'off', 'on', 'out', 'outside',
             'over', 'part', 'right', 'rightmost', 'row', 'side', 'smaller', 'top', 'towards',
             'under', 'up', 'upper', 'with'],
"other": ['small', 'little', 'long', 'large', 'largest', 'big', 'tall', 'smaller', 'bigger', 'biggest', 'tallest',
           'metal', 'silver', 'wood', 'wooden', 'plastic', 'striped', 'liquid',
           'circle', 'rectangle', 'round', 'shape', 'square', 'triangle']
}

category_weight = {
    'q_object': 5.685,
    'color': 2.95,
    'location': 1.055,
    'other': 6.118
}

categories = {
    'q_object': 0,
    'color': 1,
    'location': 2,
    'other': 3
}
c_len = len(categories)

answers = {'yes': 0, 'no': 1, 'n/a': 2}
BERT_BASE_UNCASED_PATH = os.path.join(os.environ['HOME'], "dependences", "bert_base_uncased")

with open("../data/turn_cate_prob.json") as f:
    turn_cate_prob = json.load(f)
    turn_cate_prob = {int(k): v for k, v in turn_cate_prob.items()}

RANDOM_THRESHOLD = 0.5
YES_POS = answers['yes']
