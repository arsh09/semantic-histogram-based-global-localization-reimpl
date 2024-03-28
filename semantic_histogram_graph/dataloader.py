
import cv2 
import numpy as np
import pandas as pd
import os 
from scipy.spatial.transform import Rotation

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
        else: 
            print (f"[Error] file does not exists: ")
