from data import Obj, ProjectionGenerator, Projection, Node
import numpy as np
import math
import scipy.misc as m
from numba import jit
import torch

# For rendering OBBS
from math_utils.OBB import OBB
from math_utils import Transform

class TopDownView():
    """
    Take a room, pre-render top-down views
    Of floor, walls and individual objects
    That can be used to generate the multi-channel views used in our pipeline
    """
    #Padding to avoid problems with boundary cases
    def __init__(self, height_cap=4.05, length_cap=6.05, size=512):
        #Padding by 0.05m to avoid problems with boundary cases
        """
        Parameters
        ----------
        height_cap (int): the maximum height (in meters) of rooms allowed, which will be rendered with
            a value of 1 in the depth channel. To separate the floor from empty spaces,
            floors will have a height of 0.5m. See zpad below
        length_cap (int): the maximum length/width of rooms allowed.
        size (int): size of the rendered top-down image
        Returns
        -------
        visualization (Image): a visualization of the rendered room, which is
            simply the superimposition of all the rendered parts
        (floor, wall, nodes) (Triple[torch.Tensor, torch.Tensor, list[torch.Tensor]):
            rendered invidiual parts of the room, as 2D torch tensors
            this is the part used by the pipeline
        """
        self.size = size
        self.pgen = ProjectionGenerator(room_size_cap=(length_cap, height_cap, length_cap), \
                                        zpad=0.5, img_size=size)

    def render(self, room):
        projection = self.pgen.get_projection(room)
        
        visualization = np.zeros((self.size,self.size))
        nodes = []

        for node in room.nodes:
            modelId = node.modelId #Camelcase due to original json

            t = np.asarray(node.transform).reshape(4,4)

            o = Obj(modelId)
            t = projection.to_2d(t)
            o.transform(t)
            
            save_t = t
            t = projection.to_2d()
            bbox_min = np.dot(np.asarray([node.xmin, node.zmin, node.ymin, 1]), t)
            bbox_max = np.dot(np.asarray([node.xmax, node.zmax, node.ymax, 1]), t)
            xmin = math.floor(bbox_min[0])
            ymin = math.floor(bbox_min[2])
            xsize = math.ceil(bbox_max[0]) - xmin + 1
            ysize = math.ceil(bbox_max[2]) - ymin + 1

            description = {}
            description["modelId"] = modelId
            description["transform"] = node.transform
            description["bbox_min"] = bbox_min
            description["bbox_max"] = bbox_max
            description["id"] = node.id
            description["child"] = [c.id for c in node.child] if node.child else None
            description["parent"] = node.parent.id if isinstance(node.parent, Node) else node.parent
            #if description["parent"] is None or description["parent"] == "Ceiling":
            #    print(description["modelId"])
            #    print(description["parent"])
            #    print(node.zmin - room.zmin)
                #print("FATAL ERROR")
            
            #Since it is possible that the bounding box information of a room
            #Was calculated without some doors/windows
            #We need to handle these cases
            if ymin < 0: 
                ymin = 0
            if xmin < 0: 
                xmin = 0

            #xmin = 0
            #ymin = 0
            #xsize = 256
            #ysize = 256
            
            #print(list(bbox_min), list(bbox_max))
            #print(xmin, ymin, xsize, ysize)
            rendered = self.render_object(o, xmin, ymin, xsize, ysize, self.size)
            description["height_map"] = torch.from_numpy(rendered).float()

            #tmp = np.zeros((self.size, self.size))
            #tmp[xmin:xmin+rendered.shape[0],ymin:ymin+rendered.shape[1]] = rendered
            #visualization += tmp

            # Compute the pixel-space dimensions of the object before it has been
            #    transformed (i.e. in object space)
            objspace_bbox_min = np.dot(o.bbox_min, t)
            objspace_bbox_max = np.dot(o.bbox_max, t)
            description['objspace_dims'] = np.array([
                objspace_bbox_max[0] - objspace_bbox_min[0],
                objspace_bbox_max[2] - objspace_bbox_min[2]
            ])

            # Render an OBB height map as well
            bbox_dims = o.bbox_max - o.bbox_min
            model_matrix = Transform(scale=bbox_dims[:3], translation=o.bbox_min[:3]).as_mat4()
            full_matrix = np.matmul(np.transpose(save_t), model_matrix)
            obb = OBB.from_local2world_transform(full_matrix)
            obb_tris = np.asarray(obb.get_triangles(), dtype=np.float32)
            bbox_min = np.min(np.min(obb_tris, 0), 0)
            bbox_max = np.max(np.max(obb_tris, 0), 0)
            xmin, ymin = math.floor(bbox_min[0]), math.floor(bbox_min[2])
            xsize, ysize = math.ceil(bbox_max[0]) - xmin + 1, math.ceil(bbox_max[2]) - ymin + 1
            rendered_obb = self.render_object_helper(obb_tris, xmin, ymin, xsize, ysize, self.size)
            description["height_map_obb"] = torch.from_numpy(rendered_obb).float()
            description['bbox_min_obb'] = bbox_min
            description['bbox_max_obb'] = bbox_max

            tmp = np.zeros((self.size, self.size))
            tmp[xmin:xmin+rendered.shape[0],ymin:ymin+rendered.shape[1]] = rendered
            visualization += tmp

            nodes.append(description)
        
        if hasattr(room, "transform"):
            t = projection.to_2d(np.asarray(room.transform).reshape(4,4))
        else:
            t = projection.to_2d()
        #Render the floor 
        o = Obj(room.modelId+"f", room.house_id, is_room=True)
        o.transform(t)
        floor = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += floor
        floor = torch.from_numpy(floor).float()
        
        #Render the walls
        o = Obj(room.modelId+"w", room.house_id, is_room=True)
        o.transform(t)
        wall = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += wall
        wall = torch.from_numpy(wall).float()
        return (visualization, (floor, wall, nodes))
    
    @staticmethod
    @jit(nopython=True)
    def render_object_helper(triangles, xmin, ymin, xsize, ysize, img_size):
        result = np.zeros((img_size, img_size), dtype=np.float32)
        N, _, _ = triangles.shape

        for triangle in range(N):
            x0,z0,y0 = triangles[triangle][0]
            x1,z1,y1 = triangles[triangle][1]
            x2,z2,y2 = triangles[triangle][2]
            a = -y1*x2 + y0*(-x1+x2) + x0*(y1-y2) + x1*y2
            if a != 0:
                for i in range(max(0,math.floor(min(x0,x1,x2))), \
                               min(img_size,math.ceil(max(x0,x1,x2)))):
                    for j in range(max(0,math.floor(min(y0,y1,y2))), \
                                   min(img_size,math.ceil(max(y0,y1,y2)))):
                        x = i+0.5
                        y = j+0.5
                        s = (y0*x2 - x0*y2 + (y2-y0)*x + (x0-x2)*y)/a
                        t = (x0*y1 - y0*x1 + (y0-y1)*x + (x1-x0)*y)/a
                        if s < 0 and t < 0:
                            s = -s
                            t = -t
                        if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                            height = z0 *(1-s-t) + z1*s + z2*t
                            result[i][j] = max(result[i][j], height)

        return result[xmin:xmin+xsize, ymin:ymin+ysize]

    @staticmethod
    def render_object(o, xmin, ymin, xsize, ysize, img_size):
        """
        Render a cropped top-down view of object
        
        Parameters
        ----------
        o (list[triple]): object to be rendered, represented as a triangle mesh
        xmin, ymin (int): min coordinates of the bounding box containing the object,
            with respect to the full image
        xsize, ysze (int); size of the bounding box containing the object
        img_size (int): size of the full image
        """
        triangles = np.asarray(list(o.get_triangles()), dtype=np.float32)
        return TopDownView.render_object_helper(triangles, xmin, ymin, xsize, ysize, img_size)
        
    @staticmethod
    @jit(nopython=True)
    def render_object_full_size_helper(triangles, size):
        result = np.zeros((size, size), dtype=np.float32)
        N, _, _ = triangles.shape

        for triangle in range(N):
            x0,z0,y0 = triangles[triangle][0]
            x1,z1,y1 = triangles[triangle][1]
            x2,z2,y2 = triangles[triangle][2]
            a = -y1*x2 + y0*(-x1+x2) + x0*(y1-y2) + x1*y2
            if a != 0:
                for i in range(max(0,math.floor(min(x0,x1,x2))), \
                               min(size,math.ceil(max(x0,x1,x2)))):
                    for j in range(max(0,math.floor(min(y0,y1,y2))), \
                                   min(size,math.ceil(max(y0,y1,y2)))):
                        x = i+0.5
                        y = j+0.5
                        s = (y0*x2 - x0*y2 + (y2-y0)*x + (x0-x2)*y)/a
                        t = (x0*y1 - y0*x1 + (y0-y1)*x + (x1-x0)*y)/a
                        if s < 0 and t < 0:
                            s = -s
                            t = -t
                        if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                            height = z0 *(1-s-t) + z1*s + z2*t
                            result[i][j] = max(result[i][j], height)
        
        return result

    @staticmethod
    def render_object_full_size(o, size):
        """
        Render a full-sized top-down view of the object, see render_object
        """
        triangles = np.asarray(list(o.get_triangles()), dtype=np.float32)
        return TopDownView.render_object_full_size_helper(triangles, size)

if __name__ == "__main__":
    from .house import House
    h = House(id_="51515da17cd4b575775cea4f5546737a")
    r = h.rooms[0]
    renderer = TopDownView()
    img = renderer.render(r)[0]
    img = m.toimage(img, cmin=0, cmax=1)
    img.show()
