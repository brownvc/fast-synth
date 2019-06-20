import sys, os
from data import *
import random
import scipy.misc as m
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import *
from torch.autograd import Variable
from PIL import Image
import copy
from model_prior import *
from support_prior import *
from priors.observations import ObjectCollection
from utils import stdout_redirected
from dims import Model as DimsModel
from orient import Model as OrientModel
from loc import Model as LocationModel
from cat import NextCategory as CategoryModel
from math_utils.OBB import OBB
from math_utils import Transform
from filters.global_category_filter import *

model_root_dir = '.'

model_dir = "train"

model_dir = f'{model_root_dir}/{model_dir}'

cat_dir = f"{model_dir}/bedroom/nextcat_30.pt"

loc_dir = f"{model_dir}/bedroom/location_150.pt"

orient_dir = f"{model_dir}/bedroom/orient_500.pt"

dims_dir = f"{model_dir}/bedroom/dims_200.pt"

data_dir = "bedroom_6x6"

save_dir = "release_test"
seed = 60
temperature_pixel = 0.8
temperature_cat = 1

save_vis = False
scene_indices = range(5000,5001)
trials = 1

def run_full_synth():
    """
    Synthesis from scratch. Take a range of indices, grab those scenes from the dataset
    Remove any objects from those rooms (if any), and run synthesis on the resulting empty rooms
    """
    a = SceneSynth(data_dir=data_dir)

    save_vis = False
    scene_indices = range(0, 800)
    trials = 1

    a.synth(scene_indices, trials=trials, save_dir=save_dir, temperature_pixel=temperature_pixel, \
            seed=seed, save_visualizations=save_vis)

def run_scene_completion():
    """
    Similar to run_full_synth, but instead of removing all objects,
    leave some in the room so we start from a partial scene
    """
    #Not tested fully on this version of the code
    from latent_dataset import LatentDataset
    save_vis = False
    train_size = 5000
    test_size = 20
    trials = 10
    dataset = LatentDataset(
        data_folder = data_dir,
        scene_indices = (train_size, train_size + test_size),
        importance_order = True
    )
    synth = SceneSynth(data_dir=data_dir)
    for scene_i in range(test_size):
        # Grab a random scene, sort and slice its objects, then
        #    run multiple completions onf it
        scene, input_nodes, output_node = dataset.get_scene(scene_i)
        scene.object_nodes = input_nodes
        synth.complete_room(scene, save_dir, trials)

def run_object_suggestion():
    """
    Partial scene completion, but only runs a single step of synthesis
    """
    #Not tested fully on this version of the code
    from latent_dataset import LatentDataset
    save_vis = False
    train_size = 5000
    test_size = 20
    trials = 10
    dataset = LatentDataset(
        data_folder = data_dir,
        scene_indices = (train_size, train_size + test_size),
        importance_order = True
    )
    synth = SceneSynth(data_dir=data_dir)
    for scene_i in range(test_size):
        # Grab a random scene, sort and slice its objects, then
        #    run multiple completions onf it
        scene, input_nodes, output_node = dataset.get_scene(scene_i)
        scene.object_nodes = input_nodes
        synth.suggest_next_object(scene, save_dir, trials)

class SceneSynth():
    """
    Class that synthesizes scenes
    based on the trained models
    """
    def __init__(self, data_dir, data_root_dir=None, size=256):
        """
        Parameters
        ----------
        location_epoch, rotation_epoch, continue_epoch (int):
            the epoch number of the respective trained models to be loaded
        data_dir (string): location of the dataset relative to data_root_dir
        data_root_dir (string or None, optional): if not set, use the default data location,
            see utils.get_data_root_dir
        size (int): size of the input image
        """
        Node.warning = False
        self.data_dir_relative = data_dir #For use in RenderedScene
        if not data_root_dir:
            self.data_root_dir = utils.get_data_root_dir()
        self.data_dir = f"{self.data_root_dir}/{data_dir}"
        
        #Loads category and model information
        self.categories, self.cat_to_index = self._load_category_map()
        #Hardcode second tier here because we don't have planit yet
        self.second_tiers = ["table_lamp",
                             "television",
                             "picture_frame",
                             "books",
                             "book",
                             "laptop",
                             "vase",
                             "plant",
                             "console",
                             "stereo_set",
                             "toy",
                             "fish_tank",
                             "cup",
                             "glass",
                             "fruit_bowl",
                             "bottle",
                             "fishbowl",
                             "pillow",
                             ]
        self.num_categories = len(self.categories)

        #Misc Handling
        self.pgen = ProjectionGenerator()

        self.possible_models = self._load_possible_models()
        self.model_set_list = self._load_model_set_list()
        
        #Loads trained models and build up NNs
        self.model_cat = self._load_category_model()
        self.model_location, self.fc_location = self._load_location_model()
        self.model_orient = self._load_orient_model()
        self.model_dims = self._load_dims_model()

        self.softmax = nn.Softmax(dim=1)
        self.softmax.cuda()
        
        self.model_sampler = ModelPrior()
        self.model_sampler.load(self.data_dir)

        sp = SupportPrior()
        sp.load(self.data_dir)
        self.possible_supports = sp.possible_supports

        self.obj_data = ObjectData()
        self.object_collection = ObjectCollection()

    
    def _load_category_map(self):
        with open(f"{self.data_dir}/final_categories_frequency", "r") as f:
            lines = f.readlines()
        cats = [line.split()[0] for line in lines]
        categories = [cat for cat in cats if cat not in set(['window', 'door'])]
        cat_to_index = {categories[i]:i for i in range(len(categories))}

        return categories, cat_to_index

    def _load_category_model(self, cat_dir=cat_dir):
        model_cat = CategoryModel(self.num_categories+8, self.num_categories, 200)
        model_cat.load_state_dict(torch.load(cat_dir))
        model_cat.eval()
        model_cat.cuda()

        return model_cat
    
    def _load_location_model(self, loc_dir=loc_dir):
        model_location = LocationModel(num_classes=self.num_categories+1, num_input_channels=self.num_categories+8)
        model_location.load_state_dict(torch.load(loc_dir))
        model_location.eval()
        model_location.cuda()

        return model_location, None
    
    def _load_orient_model(self, orient_dir=orient_dir):
        model_orient = OrientModel(10, 40, self.num_categories+8)
        model_orient.load(orient_dir)
        model_orient.eval()
        model_orient.snapping = True
        model_orient.testing = True
        model_orient.cuda()
        return model_orient

    def _load_dims_model(self, dims_dir=dims_dir):
        model_dims = DimsModel(10, 40, self.num_categories+8)
        model_dims.load(dims_dir)
        model_dims.eval()
        model_dims.cuda()
        return model_dims

    def _load_possible_models(self, model_freq_threshold=0):
        #model_freq_threshold: discards models with frequency less than the threshold
        category_dict = ObjectCategories()
        possible_models = [[] for i in range(self.num_categories)]
        with open(f"{self.data_dir}/model_frequency") as f:
            models = f.readlines()

        with open(f"{self.data_dir}/model_dims.pkl", 'rb') as f:
            dims = pickle.load(f)

        models = [l[:-1].split(" ") for l in models]
        models = [(l[0], int(l[1])) for l in models]
        for model in models:
            category = category_dict.get_final_category(model[0])
            if not category in ["door", "window"]:
                possible_models[self.cat_to_index[category]].append(model)

        for i in range(self.num_categories):
            total_freq = sum([a[1] for a in possible_models[i]])
            possible_models[i] = [(a[0], dims[a[0]]) for a in possible_models[i] if a[1]/total_freq > model_freq_threshold]

        return possible_models
    
    def _load_model_set_list(self):
        possible_models = self.possible_models
        obj_data = ObjectData()
        model_set_list = [None for i in range(self.num_categories)]
        for category in range(self.num_categories):
            tmp_dict = {}
            for model in possible_models[category]:
                setIds = [a for a in obj_data.get_setIds(model[0]) if a != '']
                for setId in setIds:
                    if setId in tmp_dict:
                        tmp_dict[setId].append(model[0])
                    else:
                        tmp_dict[setId] = [model[0]]
            model_set_list[category] = \
                [value for key,value in tmp_dict.items() if len(value) > 1]
        
        return model_set_list

    def get_relevant_models(self, category, modelId):
        """
        Given a category and a modelId, return all models that are relevant to it
        Which is: the mirrored version of the model,
        plus all the models that belong to the same model set
        that appear more than model_freq_threshold (set to 0.01)
        See _load_possible_models and _load_model_set_list

        Parameters
        ----------
        category (int): category of the object
        modelId (String): modelId of the object

        Return
        ------
        set[String]: set of all relevant modelIds
        """
        relevant = set()
        if "_mirror" in modelId:
            mirrored = modelId.replace("_mirror", "")
        else:
            mirrored = modelId + "_mirror"
        if mirrored in self.possible_models[category]:
            relevant.add(mirrored)

        for model_set in self.model_set_list[category]:
            if modelId in model_set:
                relevant |= set(model_set)
        
        return relevant

    def is_second_tier(self, category):
        return self.categories[category] in self.second_tiers

    def synth(self, room_ids, trials=1, size=256, samples=64, save_dir=".", \
              temperature_cat=0.25, temperature_pixel=0.4, min_p=0.5, max_collision=-0.1, seed=seed, \
              save_visualizations=False):
        """
        Synthesizes the rooms!

        Parameters
        ----------
        room_ids (list[int]): indices of the room to be synthesized, loads their
            room arthicture, plus doors and windows, and synthesize the rest
        trials (int): number of layouts to synthesize per room
        size (int): size of the top-down image
        samples (int): size of the sample grid (for location and category)
        save_dir (str): location where the synthesized rooms are saved
        temperature_cat, temperature_pixel (float): temperature for tempering,
            refer to the paper for more details
        min_p (float): minimum probability where a model instance + orientation can be accepted
        max_collision (float): max number of collision penetration, in meters, that are allowed to occur
            This is not the only collision criteria, two more are hard coded, see SynthedRoom._get_collisions
        """
        for room_id in room_ids:
            if seed:
                random.seed(seed+room_id)
            for trial in range(trials):
                if seed:
                    cur_seed = random.randint(0,10000)
                else:
                    cur_seed = None
                self.synth_room(room_id, trial, size, samples, save_dir, temperature_cat, temperature_pixel, min_p, max_collision, cur_seed, save_visualizations)

    def synth_room(self, room_id, trial, size, samples, \
                   save_dir, temperature_cat, temperature_pixel, \
                   min_p, max_collision, seed=None, save_visualizations=False):
        """
        Synthesize a single room, see synth for explanation of some most paramters
        """
        room = SynthedRoom(room_id, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision, seed, save_dir, save_visualizations)

        room.synthesize()

    # Do partial scene completion
    def complete_room(self, scene, save_dir, trials=1, size=256, samples=64, temperature_cat=0.25, temperature_pixel=0.4, \
                      min_p=0.5, max_collision=-0.1, seed=seed, save_visualizations=False):
        random.seed(seed + scene.index)
        for trial in range(trials):
            cur_seed = random.randint(0,10000)
            room = SynthedRoom(scene, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision, cur_seed, save_dir, save_visualizations, save_initial_state=True)
            # random.seed(random.randint(0,100000))
            # torch.manual_seed(random.randint(0,100000))
            room.synthesize()

    # Do next-object-suggestion
    def suggest_next_object(self, scene, save_dir, trials=1, size=256, samples=64, temperature_cat=0.25, temperature_pixel=0.4, \
                      min_p=0.5, max_collision=-0.1, seed=seed, save_visualizations=False):
        random.seed(seed + scene.index)
        for trial in range(trials):
            cur_seed = random.randint(0,10000)
            room = SynthedRoom(scene, trial, size, samples, self, temperature_cat, temperature_pixel, min_p, max_collision, cur_seed, save_dir, save_visualizations, save_initial_state=True)
            # random.seed(random.randint(0,100000))
            # torch.manual_seed(random.randint(0,100000))
            room.synthesize(num_steps=1)

class SynthedRoom():
    """
    Class that synthesize a single room and keeps its record
    """

    def __init__(self, room_id_or_room, trial, size, samples, synthesizer, temperature_cat,
                 temperature_pixel, min_p, max_collision, seed, save_dir, save_visualizations,
                 save_initial_state=False):
        """
        Refer to SceneSynth.synth for explanations for most parameters

        Parameters
        ----------
        synthesizer (SceneSynth): links back to SceneSynth so we can use the loaded models
        """
        self.__dict__.update(locals())
        del self.self #Of course I don't care about readability
        del self.room_id_or_room

        starting_with_partial_scene = (not isinstance(room_id_or_room, int))

        if not starting_with_partial_scene:
            self.room_id = room_id_or_room
            self.scene = RenderedScene(index = self.room_id, \
                                       data_dir = synthesizer.data_dir_relative, \
                                       data_root_dir = synthesizer.data_root_dir, \
                                       load_objects = False)
        else:
            room = room_id_or_room
            self.scene = room
            self.room_id = room.index

        self.composite = self.scene.create_composite()
        self.composite_obb = self.scene.create_composite()
        if starting_with_partial_scene:
            self.composite.add_nodes(self.scene.object_nodes)
            self.composite_obb.add_nodes_obb(self.scene.object_nodes)
        self.floor_top = 0.1209 #hard code as that seems to be a constant
        self.door_window_nodes = self.scene.door_window_nodes
        if starting_with_partial_scene:
            self.object_nodes = [SynthNode.fromDatasetNode(self, node) for node in self.scene.object_nodes]
        else:
            self.object_nodes = []
        self.empty_house_json = None
        self.failures = 0
        self.seed = seed
        self.save_dir = save_dir
        self.save_visualizations = save_visualizations

    def synthesize(self, num_steps=100000000):
        if self.seed:
            random.seed(self.seed)
            torch.manual_seed(random.randint(0,100000))
        print("New Room")
        failure = 0
        self.generate_current_composites()
        self.support_map = torch.zeros_like(self.current_room[3]) - 1
        self.support_map[self.current_room[0]==1] = self.synthesizer.num_categories + 1
        if self.save_initial_state:
            self.save_top_down_view()
            self.save_json()
        for i in range(num_steps):
            print(f'*** Synth step {len(self.object_nodes)}')
            self.generate_current_composites()
            if self.save_visualizations:
                self.save_top_down_view()
                self.save_json()
            next_cat = self.sample_next_cat()
            if (next_cat != self.synthesizer.num_categories) and \
               len(self.object_nodes) < 30: #Prevent extreme failures I suppose
                    success, x, y, z, sin, cos, modelId = self.sample_everything_else(next_cat)
                    if not success:
                        print("Failed to find a reasonable location, resampling next cat")
                        failure += 1
                    else:
                        self.add_node(next_cat, modelId, x, y, z, sin, cos)
            else:
                break
            if failure > 3:
                break

        self.generate_current_composites()
        self.save_top_down_view(final=True)
        self.save_json(final=True)
    
    def generate_current_composites(self):
        self.current_room = self.composite.get_composite(num_extra_channels=0)
        self.current_room_obb = self.composite_obb.get_composite(num_extra_channels=0)

    def add_node(self, category, modelId, x, y, z, sin, cos):
        new_obj = SynthNode(modelId, category, x, y, z, sin, cos, self)
        render = new_obj.get_render()
        self.support_map[render>0] = category
        self.composite.add_height_map(render, category, sin, cos)
        self.composite_obb.add_height_map(new_obj.get_render_obb(), category, sin, cos)
        self.object_nodes.append(new_obj)

    def curr_top_down_view_filename(self, final=False):
        if final:
            return f"{self.save_dir}/{self.room_id}_{self.trial}_{len(self.object_nodes)}_final_{self.failures}.png"
        else:
            return f"{self.save_dir}/{self.room_id}_{self.trial}_{len(self.object_nodes)}.png"

    def curr_json_filename(self, final=False):
        if final:
            return f"{self.save_dir}/{self.room_id}_{self.trial}_{len(self.object_nodes)}_final_{self.failures}.json"
        else:
            return f"{self.save_dir}/{self.room_id}_{self.trial}_{len(self.object_nodes)}.json"

    def sample_next_cat(self):
        #print(len(self.object_nodes))
        synthesizer = self.synthesizer
        with torch.no_grad():
            inputs = self.current_room_obb.unsqueeze(0).cuda()
            cats = self._get_existing_categories().unsqueeze(0).cuda()
            cat, logits = self.synthesizer.model_cat.sample(inputs, cats, return_logits=True)
            cat = int(cat[0])
            self.catlogits = logits[0]  # Save for visualization later
            return cat

    def sample_everything_else(self, category):
        self.location_map = None
        #Get existing collisions, which should not be considered later
        self.existing_collisions = self._get_collisions()
        #Info about best insertion so far
        num_trials = 0
        while True:
            x,y = self._sample_location(category)
            w = 256
            x_ = ((x / w) - 0.5) * 2
            y_ = ((y / w) - 0.5) * 2

            loc = torch.Tensor([x_, y_]).unsqueeze(0).cuda()
                
            orient = torch.Tensor([math.cos(0), math.sin(0)]).unsqueeze(0).cuda()
            input_img = self.current_room_obb.unsqueeze(0).cuda()
            input_img_orient = self.inverse_xform_img(input_img, loc, orient, 64)
            noise = torch.randn(1, 10).cuda()
            orient = self.synthesizer.model_orient.generate(noise, input_img_orient, category)

            sin, cos = float(orient[0][1]), float(orient[0][0])

            input_img_dims = self.inverse_xform_img(input_img, loc, orient, 64)
            noise = torch.randn(1, 10).cuda()
            dims = self.synthesizer.model_dims.generate(noise, input_img_dims, category)
            dims_numpy = dims.detach().cpu().numpy()[0,::-1]
            
            #Model Matching hacks
            modelIds = self.synthesizer.possible_models[category]
            scores = []
            for (modelId, dims_gt) in modelIds:
                l2 = (dims_gt[0]-dims_numpy[0])**2 + (dims_gt[1]-dims_numpy[1])**2
                scores.append((modelId, l2))
            important = []
            others = []
            for node in self.object_nodes:
                if node.category == category:
                    important.append(node.modelId)
                elif ((node.x - x) ** 2 + (node.y - y) ** 2) < 2500:
                    others.append(node.modelId)
            models = self.synthesizer.model_sampler.get_models(category, important, others)
            set_augmented_models = set(models)
            for modelId in models:
                set_augmented_models |= self.synthesizer.get_relevant_models(category, modelId)
            set_augmented_models = list(set_augmented_models)

            tolerated = (dims_numpy[0]**2 + dims_numpy[1]**2) * 0.01
            scores = sorted(scores, key=lambda x:x[1])
            possible = [s for s in scores if s[1] < tolerated and s[0] in set_augmented_models]
            if len(possible) > 0:
                best_modelId = possible[0][0]
            else:
                best_modelId = scores[0][0]
                
            if self.synthesizer.is_second_tier(category):
                z = self.current_room[3][math.floor(x)][math.floor(y)]
            else:
                z = self.floor_top
            new_node = SynthNode(best_modelId, category, x, y, z, sin, cos, self)

            overhang = False
            if self.synthesizer.is_second_tier(category):
                render = new_node.get_render()
                render[render>0] = 1
                render2 = render.clone()
                render2[self.current_room[3] < z-0.01] = 0
                if render2.sum() < render.sum() * 0.7:
                    overhang = True

            if not overhang:
                collisions = self._get_collisions([new_node])
                if (len(collisions) - len(self.existing_collisions)) <= 0:
                    break

            num_trials += 1

            if num_trials > 20:
                return False, None, None, None, None, None, None

        return True, x, y, z, sin, cos, best_modelId

    def inverse_xform_img(self, img, loc, orient, output_size):
        batch_size = img.shape[0]
        matrices = torch.zeros(batch_size, 2, 3).cuda()
        cos = orient[:, 0]
        sin = orient[:, 1]
        matrices[:, 0, 0] = cos
        matrices[:, 1, 1] = cos
        matrices[:, 0, 1] = -sin
        matrices[:, 1, 0] = sin
        matrices[:, 0, 2] = loc[:, 1]
        matrices[:, 1, 2] = loc[:, 0]
        out_size = torch.Size((batch_size, img.shape[1], output_size, output_size))
        grid = F.affine_grid(matrices, out_size)
        return F.grid_sample(img, grid)

    def _get_existing_categories(self):
        #Category count to be used by networks
        existing_categories = torch.zeros(self.synthesizer.num_categories)
        for node in self.object_nodes:
            existing_categories[node.category] += 1
        return existing_categories

    def _get_category_name(self, index):
        #Return name of the category, given index
        return self.synthesizer.categories[index]

    def _sample_location(self, category):
        #Creates location category map if it haven't been created
        #Otherwise just sample from the existing one
        if self.location_map is None:
            self.location_map = self._create_location_map(category)
        
        loc = int(torch.distributions.Categorical(probs=self.location_map.view(-1)).sample())
        x,y = loc//256,loc%256
        #Clears sampled location so it does not get resampled again
        self.location_map[x][y] = 0

        return x+0.5,y+0.5

    def _create_location_map(self, category):
        synthesizer = self.synthesizer
        num_categories = synthesizer.num_categories
        size = self.size

        inputs = self.current_room.unsqueeze(0)
        with torch.no_grad():
            inputs = inputs.cuda()
            outputs = synthesizer.model_location(inputs)
            outputs = synthesizer.softmax(outputs)
            outputs = F.upsample(outputs, mode='bilinear', scale_factor=4).squeeze()[category+1]
            outputs[self.current_room[0] == 0] = 0
            outputs[self.current_room[1] > 0] = 0

            location_map = outputs.cpu()
            #print(location_map)
        
        #Restricts second tier placement on possible supporting surfaces
        #If no possible options just yolo
        if self.synthesizer.is_second_tier(category):
            support_mask = torch.zeros_like(self.support_map)
            possible_supports = self.synthesizer.possible_supports[category]
            for support in possible_supports:
                support_mask[self.support_map==support] = 1
            if support_mask.sum() > 0: #Otherwise there's no possible support surface and whatever
                location_map = location_map * support_mask
        location_map = location_map**(1/self.temperature_pixel)
        location_map = location_map / location_map.sum()

        return location_map
    
    def _get_collisions(self, additional_nodes=None):
        with stdout_redirected():
            oc = self.synthesizer.object_collection
            oc.reset()
            oc.init_from_house(House(house_json=self.get_json(additional_nodes)))
            contacts = oc.get_collisions(include_collision_with_static=True)
            collisions = []
        for (_, contact_record) in contacts.items():
            #print(collision_pair)
            if contact_record.idA != contact_record.idB:
                #If contact with the room geometry, be more lenient and allow anything with 
                #less than 0.25m overlap
                if "0_0" in contact_record.idA or "0_0" in contact_record.idB:
                    if contact_record.distance < -0.1:
                        collisions.append(contact_record)
                else:
                    #Else, check if collision amount is more than max_collision, if, then it is a collision
                    if contact_record.distance < self.max_collision:
                        collisions.append(contact_record)
                    #Else, we do an additional check to see overlap of two objects along either axes
                    #Are greater than 1/5 of the smaller object
                    #Just a rough heuristics to make sure small objects don't overlap too much
                    #Since max_collision can be too large for those
                    elif contact_record.distance < -0.05:
                        idA = contact_record.idA
                        idB = contact_record.idB
                        aabbA = oc._objects[idA].obb.to_aabb()
                        aabbB = oc._objects[idB].obb.to_aabb()
                        def check_overlap(amin,amax,bmin,bmax):
                            return max(0, min(amax, bmax) - max(amin, bmin))
                        x_overlap = check_overlap(aabbA[0][0],aabbA[1][0],aabbB[0][0],aabbB[1][0])
                        y_overlap = check_overlap(aabbA[0][2],aabbA[1][2],aabbB[0][2],aabbB[1][2])
                        x_overlap /= min((aabbA[1][0]-aabbA[0][0]), (aabbB[1][0]-aabbB[0][0]))
                        y_overlap /= min((aabbA[1][2]-aabbA[0][2]), (aabbB[1][2]-aabbB[0][2]))
                        if (x_overlap > 0.2 and y_overlap > 0.2):
                            collisions.append(contact_record)

        return collisions

    def save_top_down_view(self, final=False):
        """
        Save the top down view of the current room

        Parameters
        ----------
        save_dir (String): location to be saved
        final (bool, optional): If true, mark the saved one as finalized
            and include number of failures, to be processed by other scripts
        """
        #current_room_obb = self.composite_obb.get_composite(num_extra_channels=0)
        img = m.toimage(self.current_room[3].numpy(), cmin=0, cmax=1)
        #img_obb = m.toimage(current_room_obb[3].numpy(), cmin=0, cmax=1)
        img.save(self.curr_top_down_view_filename(final))
    
    def _create_empty_house_json(self):
        #Preprocess a json containing only the empty ROOM (oops I named the method wrong)
        #Since this is constant across the entire synthesis process
        house_original = House(include_support_information=False, \
                               file_dir=f"{self.synthesizer.data_dir}/json/{self.room_id}.json")
        room_original = house_original.rooms[0] #Assume one room
        house = {}
        house["version"] = "suncg@1.0.0"
        house["id"] = house_original.id
        house["up"] = [0,1,0]
        house["front"] = [0,0,1]
        house["scaleToMeters"] = 1
        level = {}
        level["id"] = "0"
        room = {}
        room["id"] = "0_0"
        room["type"] = "Room"
        room["valid"] = 1
        room["modelId"] = room_original.modelId
        room["nodeIndices"] = []
        room["roomTypes"] = room_original.roomTypes
        room["bbox"] = room_original.bbox
        level["nodes"] = [room]
        house["levels"] = [level]
        count = 1
        for node in self.door_window_nodes:
            node_json = {}
            room["nodeIndices"].append(str(count))
            node_json["id"] = f"0_{count}"
            node_json["type"] = "Object"
            node_json["valid"] = 1
            modelId = node["modelId"]
            transform = node["transform"]
            if "mirror" in modelId:
                transform = np.asarray(transform).reshape(4,4)
                t_reflec = np.asarray([[-1, 0, 0, 0], \
                                  [0, 1, 0, 0], \
                                  [0, 0, 1, 0], \
                                  [0, 0, 0, 1]])
                transform = np.dot(t_reflec, transform)
                transform = list(transform.flatten())
                modelId = modelId.replace("_mirror","")
            node_json["transform"] = transform
            node_json["modelId"] = modelId

            level["nodes"].append(node_json)
            count += 1

        self.empty_house_json = house
        self.projection = self.synthesizer.pgen.get_projection(room_original)
        #Camera parameters, for orthographic render
        house["camera"] = {}
        ortho_param = self.projection.get_ortho_parameters()
        orthographic = {"left" : ortho_param[0],
                        "right" : ortho_param[1],
                        "bottom" : ortho_param[2],
                        "top" : ortho_param[3],
                        "far" : ortho_param[4],
                        "near" : ortho_param[5]}
        house["camera"]["orthographic"] = orthographic


    def save_json(self, final=False):
        """
        Save the json file, see save_top_down_view
        """
        house = self.get_json()
        with open(self.curr_json_filename(final), 'w') as f:
            json.dump(house, f)

    def get_json(self, additional_nodes=None):
        """
        Get the json of the current room, plus additional_nodes

        Parameters
        ----------
        additional_nodes (list[SynthNode]): objects to be included
            in addition to the current room, this is used
            to compute the collisions, since those codes are based
            on the original SUNCG json format
        """
        if self.empty_house_json is None:
            self._create_empty_house_json()

        house = copy.deepcopy(self.empty_house_json)
        level = house["levels"][0]
        room = level["nodes"][0]

        if additional_nodes is None:
            object_nodes = self.object_nodes
        else:
            object_nodes = self.object_nodes + additional_nodes

        count = len(self.door_window_nodes) + 1
        for node in object_nodes:
            node_json = {}
            room["nodeIndices"].append(str(count))
            node_json["id"] = f"0_{count}"
            node_json["type"] = "Object"
            node_json["valid"] = 1
            modelId = node.modelId
            transformation_3d = self.projection.to_3d(node.get_transformation())
            if "mirror" in modelId:
                t_reflec = np.asarray([[-1, 0, 0, 0], \
                                  [0, 1, 0, 0], \
                                  [0, 0, 1, 0], \
                                  [0, 0, 0, 1]])
                transformation_3d = np.dot(t_reflec, transformation_3d)
                modelId = modelId.replace("_mirror","")
            
            alignment_matrix = self.synthesizer.obj_data.get_alignment_matrix(modelId)
            if alignment_matrix is not None:
                transformation_3d = np.dot(alignment_matrix, transformation_3d)
            node_json["modelId"] = modelId
            node_json["transform"] = list(transformation_3d.flatten())

            level["nodes"].append(node_json)
            count += 1

        return house

class SynthNode():
    """
    Representing a node in synthesis time
    """
    def __init__(self, modelId, category, x, y, z, sin, cos, room):
        self.__dict__.update(locals())
        del self.self
        self.render = None

    @staticmethod
    def fromDatasetNode(room, node):
        """
        Convert from a node used in the training process
        """
        modelId = node['modelId']
        category = node['category']

        xmin, zmin, ymin, _ = node["bbox_min"]
        xmax, _, ymax, _ = node["bbox_max"]
        x = (xmin+xmax)/2
        y = (ymin+ymax)/2
        z = zmin

        xform = node["transform"]
        cos = xform[0]
        sin = xform[8]

        return SynthNode(modelId, category, x, y, z, sin, cos, room)
    
    def get_render(self):
        """
        Get the top-down render of the object
        """
        o = Obj(self.modelId)
        o.transform(self.get_transformation())
        render = torch.from_numpy(TopDownView.render_object_full_size(o, self.room.size))
        self.render = render
        return render

    def get_render_obb(self):
        o = Obj(self.modelId)

        bbox_dims = o.bbox_max - o.bbox_min
        model_matrix = Transform(scale=bbox_dims[:3], translation=o.bbox_min[:3]).as_mat4()
        full_matrix = np.matmul(np.transpose(self.get_transformation()), model_matrix)
        obb = OBB.from_local2world_transform(full_matrix)
        obb_tris = np.asarray(obb.get_triangles(), dtype=np.float32)
        rendered_obb = torch.from_numpy(TopDownView.render_object_full_size_helper(obb_tris, self.room.size))
        return rendered_obb

    def get_transformation(self):
        """
        Get the transformation matrix
        Used to render the object
        and to save in json files
        """
        x,y,z = self.x, self.y, self.z
        xscale = self.room.synthesizer.pgen.xscale
        yscale = self.room.synthesizer.pgen.yscale
        zscale = self.room.synthesizer.pgen.zscale
        zpad = self.room.synthesizer.pgen.zpad

        sin, cos = self.sin, self.cos

        t = np.asarray([[cos, 0, -sin, 0], \
                        [0, 1, 0, 0], \
                        [sin, 0, cos, 0], \
                        [0, 0, 0, 1]])
        t_scale = np.asarray([[xscale, 0, 0, 0], \
                              [0, zscale, 0, 0], \
                              [0, 0, xscale, 0], \
                              [0, 0, 0, 1]])
        t_shift = np.asarray([[1, 0, 0, 0], \
                              [0, 1, 0, 0], \
                              [0, 0, 1, 0], \
                              [x, z, y, 1]])
        
        return np.dot(np.dot(t,t_scale), t_shift)


if __name__ == '__main__':
    utils.ensuredir(save_dir)
    os.system(f'rm -f {save_dir}/*')

    run_full_synth()
    #run_scene_completion()
    #run_object_suggestion()
