import numpy as np 


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


