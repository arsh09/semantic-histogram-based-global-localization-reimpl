import numpy as np 
import open3d as o3d 

class GraphRegistration: 

    def __init__(self, map_nodes , query_nodes , matches ) -> None:
    
        self.map_nodes = map_nodes
        self.query_nodes = query_nodes 
        self.matches = matches

        self.source_points = []
        self.target_points = []
        self.label_vector = []
        self.ransac_result = None

        self.get_source_and_target_points()

    def get_source_and_target_points(self) : 

        rows = self.matches.shape[0]

        self.source_points = np.zeros( (rows, 3) )
        self.target_points = np.zeros( (rows, 3) )
        self.label_points = np.zeros( (rows, 3) )

        for row in range(rows): 
            
            i = int ( self.matches[row, 0] )
            j = int ( self.matches[row, 1] )

            # this condition because no-neighbor thing 
            # is not implemented!
            if i < len(self.map_nodes):
                self.source_points[row, :] = self.map_nodes[i].XYZ
                self.label_points[row] = self.map_nodes[i].label

            if j < len(self.query_nodes):
               self.target_points[row, :] = self.query_nodes[j].XYZ
  

    def match_ransac(self): 

        # Convert numpy arrays to Open3D point clouds
        source_cloud = o3d.geometry.PointCloud()
        target_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(self.source_points)
        target_cloud.points = o3d.utility.Vector3dVector(self.target_points)

        # Run RANSAC to estimate the transformation
        max_correspondence_distance = 0.01   
        ransac_n = 4   
        iterations = 1000   
        success_probability = 0.99   
        
        self.ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source_cloud, target_cloud,
            o3d.utility.Vector2iVector( self.matches ),   
            max_correspondence_distance,
            ransac_n = ransac_n,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(iterations, success_probability)
        )

        print("[LOG] initial transformation matrix:")
        print(self.ransac_result.transformation)

        print(f"[LOG] number of inliers: {len(self.ransac_result.correspondence_set)}")

        if len(self.source_points) != 0:
            print ( f"[LOG] match success: { (len(self.ransac_result.correspondence_set) / len(self.source_points)) }")

        # # You can also transform the source cloud using the estimated transformation
        # transformed_source_cloud = source_cloud.transform(self.ransac_result.transformation)

