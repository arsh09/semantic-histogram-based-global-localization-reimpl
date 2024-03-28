
import cv2 
import numpy as np 
from semantic_histogram_graph.graph import GraphBuilderUnit


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

        return good_matches_id
    

