import argparse 

import os 
import cv2 
import numpy as np 
import networkx as nx 
import pandas as pd
from scipy.spatial.transform import Rotation
import open3d as o3d
import matplotlib.pyplot as plt

import sys
np.set_printoptions(threshold=sys.maxsize)


class DataLoader: 

    def __init__(self, dir : str, start_id : int, stop_id : int, label_rgb = None) -> None:
        
        self.dir = dir
        
        if not os.path.exists(self.dir): 
            raise("Database directory does not exists") 

        self.item_count = start_id 
        self.start_id = start_id 
        self.stop_id = stop_id 

        self._handle_read_all_labels(label_rgb)
        self._handle_read_all_poses()


    def _handle_read_all_labels(self, label_rgb): 

        if label_rgb == None: 
            self.label_rgb = cv2.imread( os.path.join(self.dir, "segmentation_pallet.png") )[:, :11, :]
            self.label_rgb = cv2.cvtColor( self.label_rgb, cv2.COLOR_BGR2RGB ) 
        else: 
            self.label_rgb = label_rgb

    def _handle_read_all_poses(self): 

        self.df_poses = pd.read_csv( os.path.join(self.dir, "airsim.txt"), sep=",")
        if len(self.df_poses.columns) == 7: 
            self.df_poses.columns = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
        else: 
            raise ("Poses CSV is not formatted properly.")
        
    def _handle_get_frame_pose(self, index): 

        pose = self.df_poses.iloc[index]
        
        rotation = Rotation.from_quat([
            pose['qx'], pose['qy'], pose['qz'], pose['qw'] 
        ])

        pose_transform = np.zeros((4,4), dtype=float)
        pose_transform[:3, :3] = rotation.as_matrix()
        pose_transform[0, 3] = pose['x']
        pose_transform[1, 3] = pose['y']
        pose_transform[2, 3] = pose['z']
        pose_transform[3, 3] = 1.0
        
        return pose_transform


    def _handle_camera_pose(self): 

        frame_0_pose = self._handle_get_frame_pose(self.start_id)
        frame_i_pose = self._handle_get_frame_pose(self.item_count)
        camera_pose = np.linalg.inv(frame_0_pose) * frame_i_pose
        return frame_i_pose


    def __len__(self) -> int: 
        return self.stop_id - self.start_id 

    def __iter__(self): 
        return self 
    
    def __next__(self): 
        
        if self.item_count > self.stop_id:
            raise StopIteration 
        
        seg_path = os.path.join(self.dir, f"segmentation/segmentation_{self.item_count}.png")
        depth_path = os.path.join(self.dir, f"depth/depth_{self.item_count}.png")

        if os.path.exists( seg_path ) and os.path.exists( depth_path ): 

            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            seg_img = cv2.cvtColor( seg_img, cv2.COLOR_BGR2RGB )
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) 

            frame_pose = self._handle_get_frame_pose(self.item_count)
            camera_pose = self._handle_camera_pose()

            self.item_count += 1

            return (seg_img, depth_img, self.label_rgb, frame_pose, camera_pose)


class GraphNode: 
    def __init__(self, name, world, pixel, label_num, label_gray, label_rgb) -> None:
        
        self._node_name = name
        self._center_world = world
        self._center_pixel = pixel
        self._center_label = label_num
        self._center_label_gray = label_gray
        self._center_label_rgb = label_rgb
    
    @property
    def name (self): 
        return self._node_name 
    
    @property
    def XYZ(self): 
        return self._center_world
    
    @property
    def xy(self): 
        return self._center_pixel
    
    @property
    def label(self): 
        return self._center_label

    @property
    def label_gray(self): 
        return self._center_label_gray

    @property
    def label_rgb(self): 
        return self._center_label_rgb


class GraphEdges:

    def __init__(self, nodes ) -> None:
        self.nodes = nodes

    def get_neighbors(self): 

        edges = np.zeros( ( len(self.nodes), len(self.nodes) ) , dtype=np.uint8)
        
        for i, node_i in enumerate(self.nodes): 
            for j, node_j in enumerate(self.nodes): 
                if i != j: 
                    xi, yi, zi = node_i.XYZ
                    xj, yj, zj = node_j.XYZ
                    distance = np.sqrt( (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 ) 
                    if (distance) < 20 : 
                        edges[i, j] = 1

        return edges 


class HistogramDescriptors: 

    def __init__(self, edges, nodes) -> None:

        self.edges = edges 
        self.nodes = nodes 
        self.descriptors_vector = []
        self.no_neighbor_nodes = None
        self.calculate_descriptr()

    def calculate_descriptr(self): 

        cols, rows = self.edges.shape 
        self.no_neighbor_nodes = np.zeros( (cols, 1) )

        for i in range(cols): 
            descriptor = np.zeros( (11**3, 1) ) # neighbor of neihbor of 11.
            neighbor_count = 0
            neighbor_id_vector = []

            for j in range(rows): 
                if self.edges[i, j] == 1: 
                    neighbor_id_vector.append( j ) 
                    neighbor_count += 1

            if neighbor_count == 0: 
                self.no_neighbor_nodes[i, 0] = 1
                self.descriptors_vector.append( descriptor )
                continue 

            current_node_label = self.nodes[i].label
            for n_id in neighbor_id_vector: 
                first_neighbor_label = self.nodes[n_id].label 

                for nn_id in range(rows): 
                    second_neighbor_label = self.nodes[nn_id].label
                    if self.edges[n_id, nn_id] == 1: 
                        descriptor_index = current_node_label * pow(11, 2) + first_neighbor_label * pow(11,  1) + second_neighbor_label * pow(11, 0) 
                        descriptor[descriptor_index, 0] = descriptor[descriptor_index, 0] + 1

            self.descriptors_vector.append( descriptor )

    def draw_descriptor(self): 
        pass

    def get_descriptor(self, descriptor_id : int ): 

        if len (self.descriptors_vector) > descriptor_id:
            return self.descriptors_vector[descriptor_id]
        else: 
            print ("empty descriptor")
            return np.zeros( (11**3, 1) )



class GraphBuilderUnit: 

    def __init__(self, camera_intrinsic = None, node_prefix = "m") -> None:

        self.camera_intrinsic = camera_intrinsic
        self.max_points_for_node = 5000
        self._nodes = []
        self._edges = []

        self._descriptors = None
        self._node_name_ = 0
        self._node_prefix = node_prefix
    
    @property 
    def nodes(self): 
        return self._nodes 
    
    @property
    def edges(self): 
        if len(self._edges) == 0: 
            builder_edges = GraphEdges( self.nodes )
            self._edges = builder_edges.get_neighbors()

        return self._edges

    @property
    def descriptors(self): 
        if self._descriptors == None:
            self._descriptors = HistogramDescriptors(self.edges, self.nodes )

        return self._descriptors.descriptors_vector

    def get_descriptor(self, id): 
        self.descriptors
        return (self._descriptors.get_descriptor(id))

    def get_node_name(self):
        self._node_name_+= 1
        return self._node_prefix + str(self._node_name_)
    
    def handle_get_pointcloud(self, image_rgb, image_depth, camera_pose): 

        image_depth = (image_depth.astype(np.float32) * 100 / 255.0 )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(image_rgb),
            o3d.geometry.Image(image_depth),
            depth_scale=1000.0,   
            depth_trunc=3.0,      
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_intrinsic)
        return pcd

    def handle_pixel_to_world(self, point, world_transform): 

        x, y, z = point
        cx, cy = self.camera_intrinsic.intrinsic_matrix[0,1], self.camera_intrinsic.intrinsic_matrix[0,2]
        fx, fy = self.camera_intrinsic.intrinsic_matrix[0,0], self.camera_intrinsic.intrinsic_matrix[1,1]

        # Convert pixel coordinates to camera coordinates
        X_c = (x - cx) * z / fx
        Y_c = (y - cy) * z / fy
        Z_c = z
        point_in_current_frame = np.array([X_c, Y_c, Z_c, 1])

        # Transform to world coordinates
        point_in_world_frame = np.dot(world_transform, point_in_current_frame)
        
        # get the translation
        point_world = point_in_world_frame[:3] / point_in_world_frame[3]

        x ,y, z = point_world
        return point_world 


    def handle_label_extraction(self, image_rgb, image_depth, image_labels_rgb , frame_pose): 

        image_labels_gray = cv2.cvtColor(image_labels_rgb, cv2.COLOR_BGR2GRAY ).squeeze(0)
        image_gray = cv2.cvtColor( image_rgb, cv2.COLOR_RGB2GRAY) 

        for count, label_color in enumerate(image_labels_gray):
            image_mask = np.zeros(image_gray.shape)
            indices = np.where(image_gray == label_color)
            image_mask[ indices ] = 255
            
            depth_points = image_depth[ image_mask == 255 ] * 100 / 255.0
            y_points = indices[0]
            x_points = indices[1]

            if len(depth_points) >= self.max_points_for_node :
                pixel_mean_point = [np.mean(x) for x in [x_points, y_points, depth_points ]]
                world_mean_point = self.handle_pixel_to_world(pixel_mean_point, frame_pose)
                label_color_rgb = image_labels_rgb[:, count, :]
                
                node_name = self.get_node_name()
                node = GraphNode( node_name, world_mean_point, pixel_mean_point, count, label_color , label_color_rgb) 
                self.nodes.append( node )

                circle_centers = (int( pixel_mean_point[0] ), int( pixel_mean_point[1]) )                
                cv2.circle( image_rgb, circle_centers, 15, (255, 255, 0), 3 )
 

    def handle_frame(self, image_rgb, image_depth, labels, frame_pose, camera_pose):

        pass



class GraphDescriptorMatcher: 

    def __init__(self, map_graph : GraphBuilderUnit, query_graph : GraphBuilderUnit ) -> None:
        

        self.map_graph = map_graph
        self.query_graph = query_graph
        self.score_matrix = None

        self.get_matches()


    def get_matches(self): 

        """
            Note to self: no neighbor part is not implemented!

        """
        cols = len(self.map_graph.descriptors)
        rows = len(self.query_graph.descriptors)

        self.score_matrix = np.zeros( (rows, cols) )

        for i in range(rows): 
            for j in range(cols) :
                descriptor_i = self.query_graph.get_descriptor(i)
                descriptor_j = self.map_graph.get_descriptor(j)

                label_i = self.query_graph.nodes[i].label
                label_j = self.map_graph.nodes[j].label 

                if  label_i == label_j : 
                    
                    sum_top = np.sum ( descriptor_i * descriptor_j )
                    sum_bottom_left = np.sqrt ( np.sum( np.square( descriptor_i ) ) )
                    sum_bottom_right = np.sqrt( np.sum( np.square( descriptor_j ) ) )
                    sum_bottom = sum_bottom_left * sum_bottom_right 

                    if sum_bottom != 0.0: 
                        score = np.abs( sum_top / sum_bottom )
                        print (score)
                        self.score_matrix[i, j] = score
       
    def get_good_matches(self): 

        cols = len(self.map_graph.descriptors)
        rows = len(self.query_graph.descriptors)

        matcher_id = np.zeros( (rows, 3) )
        good_matches_id = []
        count = 0

        for i in range(rows):

            max_score = 0.0
            max_score_id = -1

            for j in range(cols):
                if self.score_matrix[i, j] > max_score:
                    max_score = self.score_matrix[i, j]
                    max_score_id = j

            if max_score < 0.5: 
                continue
            
            matcher_id[i, 1] = max_score_id
            matcher_id[i, 2] = 1
            count += 1        
        
        good_matches_id = np.zeros( (count, 2) )
        count = 0 
        for ii in range(rows): 
            if matcher_id[ii, 2] == 1: 
                good_matches_id[count, 0] = ii
                good_matches_id[count, 1] = matcher_id[ii, 1]
                count += 1

        print (self.score_matrix)
        print ( good_matches_id )


def main() : 

    # setup camera 
    width, height = 1024, 576
    fx, fy = 512, 512
    cx, cy = 512, 288
    intrinsic = o3d.camera.PinholeCameraIntrinsic( width, height, fx, fy, cx, cy)

    data_map = DataLoader( dir = "/data/airsim_2/forwardCar/", start_id=1, stop_id=35 )
    builder_map = GraphBuilderUnit(intrinsic, node_prefix = "m" )

    data_query = DataLoader( dir = "/data/airsim_2/backwardCar/", start_id=1, stop_id=5)
    builder_query = GraphBuilderUnit(intrinsic, node_prefix = "q" )

    for seg, depth, label, frame_pose, camera_pose in data_map: 
        pcd = builder_map.handle_get_pointcloud(seg, depth, frame_pose)
        builder_map.handle_label_extraction(seg, depth, label, frame_pose)
        cv2.imshow("seg", seg) 
        k = cv2.waitKey(100)
        if k == ord('q'):
            break 


    for seg, depth, label, frame_pose, camera_pose in data_query: 
        pcd = builder_query.handle_get_pointcloud(seg, depth, frame_pose)
        builder_query.handle_label_extraction(seg, depth, label, frame_pose)
        cv2.imshow("seg", seg) 
        k = cv2.waitKey(100)

        if k == ord('q'):
            break 

    map_nodes = builder_map.nodes
    map_edges = builder_map.edges 

    query_nodes = builder_query.nodes
    query_edges = builder_query.edges

    # matcher = GraphDescriptorMatcher( builder_map, builder_query ) 
    # matcher.get_good_matches()


    # map_graph = nx.Graph()
    # for node in map_nodes: 
    #     map_graph.add_node( node.name, pos = node.XYZ, color = node.label_rgb )


    cv2.destroyAllWindows()


if __name__ == "__main__": 

    main()




# node_positions = nx.spring_layout(map_graph, dim=3 , scale=100)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # Plot the nodes - alpha is scaled by "depth" automatically
# ax.scatter(*node_positions.T, s=100, ec="w")

# """Visualization options for the 3D axes."""
# # Turn gridlines off
# ax.grid(False)
# # Suppress tick labels
# for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
#     dim.set_ticks([])
# # Set axes labels
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")

# # nx.draw_networkx_nodes(map_graph, node_positions, cmap=plt.get_cmap('jet'),  node_size = 500)
# # nx.draw( map_graph)
# plt.show()

# print (map_edges)
