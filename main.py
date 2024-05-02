#! /usr/bin/env python3

import argparse 
import cv2 
import numpy as np 
import open3d as o3d


import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

from semantic_histogram_graph.dataloader import DataLoader 
from semantic_histogram_graph.graph import GraphBuilderUnit 
from semantic_histogram_graph.matcher import GraphDescriptorMatcher
from semantic_histogram_graph.registration import GraphRegistration


def main() : 

    # setup camera 
    width, height = 1024, 576
    fx, fy = 512, 512
    cx, cy = 512, 288

    # width, height = 1920, 1080 
    # fx, fy = 1057, 1057
    # cx, cy = 952, 553

    intrinsic = o3d.camera.PinholeCameraIntrinsic( width, height, fx, fy, cx, cy)

    data_map = DataLoader( dir = "/data/airsim_2/forwardCar/", start_id=1, stop_id=100 )
    builder_map = GraphBuilderUnit(intrinsic, node_prefix = "m" )

    data_query = DataLoader( dir = "/data/airsim_2/backwardCar/", start_id=40, stop_id=65)
    # builder_query = GraphBuilderUnit(intrinsic, node_prefix = "q" , max_depth_threshold = 125)
    builder_query = GraphBuilderUnit(intrinsic, node_prefix = "q" )

    print ("[LOG] buidling map graph")
    for seg, depth, label, frame_pose, camera_pose in data_map: 

        print (f"[LOG] map frame: {data_map.item_count}" )
        print ( camera_pose )        
        depth = (depth.astype(np.float32) * 100 / 255.0 )
        pcd = builder_map.handle_get_pointcloud(seg, depth, frame_pose)
        builder_map.handle_label_extraction(seg, depth, label, frame_pose)
        cv2.imshow("Image", seg) 
        k = cv2.waitKey(1)
        if k == ord('q'):
            break 
        print ("")
        print ("")
        

    print ("[LOG] buidling query graph")
    for seg, depth, label, frame_pose, camera_pose in data_query: 

        print (f"[LOG] query frame: {data_query.item_count}" )
        print ( camera_pose )        
        pcd = builder_query.handle_get_pointcloud(seg, depth, frame_pose)
        builder_query.handle_label_extraction(seg, depth, label, frame_pose)
        cv2.imshow("Image", seg) 
        k = cv2.waitKey(1)

        if k == ord('q'):
            break 

        print ("")
        print ("")

    map_nodes = builder_map.nodes
    map_edges = builder_map.edges 

    query_nodes = builder_query.nodes
    query_edges = builder_query.edges

    valid_map_edges = np.where( map_edges == 1 )
    valid_query_edges = np.where( query_edges == 1 )

    print (f"[LOG] number of nodes in map graph: {len(map_nodes)}")
    print (f"[LOG] number of edges in map graph: {len(valid_map_edges[0])}")
    print (f"[LOG] number of nodes in query graph: {len(query_nodes)}")
    print (f"[LOG] number of edges in query graph: {len(valid_query_edges[0])}")

    print ("[LOG] buidling graph descriptor matchers")
    matcher = GraphDescriptorMatcher( builder_map, builder_query ) 
    good_matches = matcher.get_good_matches()

    print (f"[LOG] finding inlier correspondances with {len(good_matches)} matches")
    registration = GraphRegistration( map_nodes, query_nodes , good_matches )
    registration.match_ransac()

    # map_graph = nx.Graph()
    # for node in map_nodes: 
    #     map_graph.add_node( node.name, pos = node.XYZ, color = node.label_rgb )

    cv2.destroyAllWindows()


if __name__ == "__main__": 

    main()

