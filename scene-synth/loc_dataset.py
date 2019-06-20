from torch.utils import data
import torch
from PIL import Image
from torch.autograd import Variable
import numpy as np
import scipy.misc as m
import random
import math
import pickle
from data import ObjectCategories, RenderedScene, RenderedComposite, House, ProjectionGenerator, DatasetToJSON, ObjectData
import copy
import utils
from collections import defaultdict
import os
from functools import cmp_to_key

"""
Dataset for predicting location
"""

class LocDataset():
    def __init__(self, scene_indices=(0,4000), data_folder="bedroom", data_root_dir=None, seed=None):
        super(LocDataset, self).__init__()
        self.category_map = ObjectCategories()
        self.seed = seed
        self.data_folder = data_folder
        self.data_root_dir = data_root_dir
        self.scene_indices = scene_indices

        data_root_dir = utils.get_data_root_dir()
        with open(f"{data_root_dir}/{data_folder}/final_categories_frequency", "r") as f:
            lines = f.readlines()
        self.n_categories = len(lines)-2  # -2 for 'window' and 'door'

    def __len__(self):
        return self.scene_indices[1]-self.scene_indices[0]

    def __getitem__(self,index):
        if self.seed:
            random.seed(self.seed)

        i = index+self.scene_indices[0]
        scene = RenderedScene(i, self.data_folder, self.data_root_dir)
        composite = scene.create_composite()

        object_nodes = scene.object_nodes

        random.shuffle(object_nodes)

        if 'parent' in object_nodes[0]:
            #print([a["category"] for a in object_nodes])
            # Make sure that all second-tier objects come *after* first tier ones
            def is_second_tier(node):
                return (node['parent'] != 'Wall') and \
                       (node['parent'] != 'Floor')
            object_nodes.sort(key = lambda node: int(is_second_tier(node)))

            # Make sure that all children come after their parents
            def cmp_parent_child(node1, node2):
                # Less than (negative): node1 is the parent of node2
                if node2['parent'] == node1['id']:
                    return -1
                # Greater than (postive): node2 is the parent of node1
                elif node1['parent'] == node2['id']:
                    return 1
                # Equal (zero): all other cases
                else:
                    return 0
            object_nodes.sort(key = cmp_to_key(cmp_parent_child))
            #print([a["category"] for a in object_nodes])
            #print("________________")

        num_objects = random.randint(0, len(object_nodes))
        #num_objects = len(object_nodes)
        num_categories = len(scene.categories)

        centroids = []
        parent_ids = ["Floor", "Wall"]
        for i in range(num_objects):
            node = object_nodes[i]
            if node["parent"] == "Wall":
                print("Massive messup!")
            composite.add_node(node)
            xsize, ysize = node["height_map"].shape
            xmin, _, ymin, _ = node["bbox_min"]
            xmax, _, ymax, _ = node["bbox_max"]
            parent_ids.append(node["id"])
        
        inputs = composite.get_composite(num_extra_channels=0)
        size = inputs.shape[1]

        for i in range(num_objects, len(object_nodes)):
            node = object_nodes[i]
            if node["parent"] == "Wall":
                print("Massive messup!")
            if node["parent"] in parent_ids:
                xsize, ysize = node["height_map"].shape
                xmin, _, ymin, _ = node["bbox_min"]
                xmax, _, ymax, _ = node["bbox_max"]
                centroids.append(((xmin+xmax)/2, (ymin+ymax)/2, node["category"]))
        
        output = torch.zeros((64,64)).long()

        for (x,y,label) in centroids:
            output[math.floor(x/4),math.floor(y/4)] = label+1

        return inputs, output

