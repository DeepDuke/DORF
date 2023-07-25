#! /usr/bin/python2.7
import struct
import numpy as np 

LABEL_TO_NAME = {
    0 : "unlabeled",
    1 : "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}

# color: BGR
LABEL_TO_COLOR = {
    0 : [0, 0, 0],
    1 : [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0]
}

COLOR_ZOO = {
    'red': np.array([255, 0, 0]),
    'yellow': np.array([255, 255, 0]), 
    'purple': np.array([128, 0, 128]), 
    'light_green': np.array([0, 128, 0]),
    'deep_green': np.array([0, 50, 32])
}

# Moving Object Color and Static Object Color (BGR)
MOVING_COLOR=[0, 0, 255]     # red
# STATIC_COLOR=[255, 255, 0]
GROUND_COLOR=[0, 255, 0]     # green
STATIC_COLOR=[255, 255, 255] # white

# Moving Object labels
MOVING_OBJECT_LABELS=(252, 253, 254, 255, 256, 257, 258, 259)


def get_rgba_color(BGR, A=255):
    """Encode color

    Args:
        BGR (_type_): [B, G, R] color channel values
        A (int, optional): Transparency \in [0 ~ 255]. Defaults to 255.
    """
    B, G, R = BGR
    rgba = struct.unpack('I', struct.pack('BBBB', B, G, R, A))[0]
    return rgba


def get_label_color(semantic_label):
    if semantic_label in MOVING_OBJECT_LABELS:
        return get_rgba_color(MOVING_COLOR)
    else:
        return get_rgba_color(STATIC_COLOR)


def get_random_color():
    """Returns a numpy array representing a random RGB color
    """
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    
    color = np.array([r, g, b])
    
    return color 

def get_moving_object_color():
    return get_rgba_color(MOVING_COLOR)

def get_static_object_color():
    return get_rgba_color(STATIC_COLOR)

def get_ground_object_color():
    return get_rgba_color(GROUND_COLOR)
