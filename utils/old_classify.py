# -*- coding: utf-8 -*-

question_category = {
    "q_object": [
        'person', 'vehicle', 'outdoor', 'animal', 'accessory', 'sports', 'kitchen', 'food', 'furniture', 'logos',
        'electronic', 'appliance', 'indoor', 'utensil', 'human', 'cloth', 'cloths', 'clothing', 'people', 'persons',
        "gingerbread", "earring", "hinge", 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'lettuce', 'ducks', 'duck', 'logo', 'truck', 'boat', 'traffic', 'light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'table', 'toilet', 'toilets', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush', 'meter', 'bear', 'cell', 'phone', 'wine', 'glass', 'racket',
        'baseball', 'glove', 'hydrant', 'drier', 'kite', 'sofa', 'fork', 'adult', 'arms', 'baby', 'bag', 'ball',
        'bananas', 'basket', 'bat', 'batter', 'bike', 'birds', 'board', 'body', 'books', 'bottles', 'box', 'boy',
        'bread', 'brush', 'building', 'bunch', 'cabinet', 'camera', 'candle', 'cap', 'carrots', 'cars', 'cart', 'case',
        'catcher', 'cell phone', 'chairs', 'child', 'chocolate', 'coat', 'coffee', 'computer', 'controller',
        'counter', 'cows', 'cupboard', 'cups', 'curtain', 'cycle', 'desk', 'device', 'dish', 'doll', 'door',
        'dress', 'driver', 'equipment', 'eyes', 'fan', 'feet', 'female', 'fence', 'fire', 'flag', 'flower', 'flowers',
        'foot', 'frame', 'fridge', 'fruit', 'girl', 'girls', 'glasses', 'guy', 'guys', 'hair drier', 'handle',
        'hands', 'hat', 'helmet', 'house', 'jacket', 'jar', 'jeans', 'kid', 'kids', 'lady', 'lamp', 'leg', 'legs',
        'luggage', 'machine', 'male', 'man', 'meat', 'men', 'mirror', 'mobile', 'monitor', 'mouth', 'mug', 'napkin',
        'pan', 'pants', 'paper', 'pen', 'picture', 'pillow', 'plant', 'plate', 'player', 'players', 'pole', 'pot',
        'purse', 'rack', 'racket', 'road', 'roof', 'screen', 'shelf', 'shelves', 'shirt', 'shoe', 'shoes', 'short',
        'shorts', 'shoulder', 'signal', 'sign', 'silverware', 'skate', 'ski', 'sky', 'snow', 'soap', 'speaker',
        'stairs', 'statue', 'stick', 'stool', 'stove', 'street', 'suit', 'sunglasses', 'suv', 'teddy', 'tennis',
        'tent', 'tomato', 'towel', 'tower', 'toy', 'traffic', 'tray', 'tree', 'trees', 't-shirt', 'ships',
        'tshirt', 'vegetable', 'vest', 'wall', 'watch', 'wheel', 'wheels', 'window', 'windows', 'woman', 'women'
    ],
    "action": [
        "thing", "having", "building", "starting", "single", "wings", "finger", "staying",
        "stocking", "ring", "wearing", "amazing", "according","advertising", "adjoining",
        "surrounding", "gingerbread", "strings", "earring", "hinge", "lighting", "ceiling",
        "dingy", "king", "ingredient", "kingdom" "clothing", "including"
    ],
    "color": [
        'white', 'red', 'black', 'blue', 'green', 'yellow', 'orange', 'brown', 'pink',
        'grey', 'gray', 'dark', 'purple', 'color', 'colored', 'colour', 'blond', 'beige', 'bright'
    ],
    "size": [
        'small', 'little', 'long', 'large', 'largest', 'big', 'tall', 'smaller', 'bigger', 'biggest', 'tallest'
    ],
    "texture": [
        'metal', 'silver', 'wood', 'wooden', 'plastic', 'striped', 'liquid'
    ],
    "shape": [
        'circle', 'rectangle', 'round', 'shape', 'square', 'triangle'
    ],
    "location": [
        '1st', '2nd', 'third', '3', '3rd', 'four', '4th', 'fourth', '5', '5th',
        'five', 'first', 'second', 'last', 'above', 'across', 'after', 'around', 'at',
        'away', 'back', 'background', 'before', 'behind', 'below', 'beside', 'between',
        'bottom', 'center', 'close', 'closer', 'closest', 'corner', 'directly', 'down',
        'edge', 'end', 'entire', 'facing', 'far', 'farthest', 'floor', 'foreground', 'from',
        'front', 'furthest', 'ground', 'hidden', 'in', 'inside', 'left', 'leftmost',
        'middle', 'near', 'nearest', 'next', 'next to', 'off', 'on', 'out', 'outside',
        'over', 'part', 'right', 'rightmost', 'row', 'side', 'smaller', 'top', 'towards',
        'under', 'up', 'upper', 'with'
    ]
}

category_weight = {
    'q_object': 2.1,
    'action': 3.0,
    'color': 1.0,
    'size': 10.0,
    'texture': 24.3,
    'shape': 59.0,
    'location': 0.40,
    'other': 2.7
}

categories = {
    'q_object': 0,
    'action': 1,
    'color': 2,
    'size': 3,
    'texture': 4,
    'shape': 5,
    'location': 6,
    'other': 7
}
