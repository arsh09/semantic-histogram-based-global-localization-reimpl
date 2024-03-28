
import open3d as o3d
import cv2 
import numpy as np 
from semantic_histogram_graph.descriptor import HistogramDescriptors


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
        self.max_node_to_node_distance = 2

    def get_neighbors(self): 

        edges = np.zeros( ( len(self.nodes), len(self.nodes) ) , dtype=np.uint8)
        
        for i, node_i in enumerate(self.nodes): 
            for j, node_j in enumerate(self.nodes): 
                if i != j: 
                    xi, yi, zi = node_i.XYZ
                    xj, yj, zj = node_j.XYZ
                    distance = np.sqrt( (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 ) 
                    if (distance) < self.max_node_to_node_distance : 
                        edges[i, j] = 1

        return edges 


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

        image_labels_gray = cv2.cvtColor(image_labels_rgb, cv2.COLOR_RGB2GRAY ).squeeze(0)
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

