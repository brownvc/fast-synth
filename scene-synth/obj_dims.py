from data import *
from utils import *
import pickle

# data_dir = "toilet_6x6"
# data_dir = "bedroom_6x6"
# data_dir = "living_6x6_aug"
data_dir = "office_6x6_aug"
data_root_dir = utils.get_data_root_dir()
img_size = 256
room_dim = 6.05

with open(f"{data_root_dir}/{data_dir}/model_frequency", "r") as f:
    models = f.readlines()

models = [l[:-1].split(" ") for l in models]
models = [l[0] for l in models]
#print(models)

model_dims = dict()

for model in models:
    o = Obj(model)
    dims = [(o.bbox_max[0] - o.bbox_min[0])/room_dim, \
            (o.bbox_max[2] - o.bbox_min[2])/room_dim]

    model_dims[model] = dims

with open(f"{data_root_dir}/{data_dir}/model_dims.pkl", 'wb') as f:
    pickle.dump(model_dims, f, pickle.HIGHEST_PROTOCOL)

