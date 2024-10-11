# Matmul Generator

every datapoint consists of two files:
- matmul_*.adj.json: the adjacency relation of the directed graph. The attribute name is the name of the node, and the array value is the name of the nodes that the node points to.
- matmul_*.attr.json: the attribute of the nodes. The attribute name is the name of the node, and the value is the attribute value. The last dimension is the expected output.
