# > [Package_Dependencies]
import pygame, math
import numpy as np
import random
# > [README] Is also being included here below the GridMod source code,
# I will definetly encourage you to read it if you havent done it allready 
# as i only will do some very brief commenting on some variables main methods and functions.
# > [matrix_manipulations()] Functions that can transform given vectors or matrices
def eye_matrix(dx=0,dy=0,dz=0):
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[dx,dy,dz,1]])
def eye_translate(vector,distance):
    unit_vector = np.hstack([unitVector(vector) * distance,1])
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],unit_vector])
def eye_scale(s,cx=0,cy=0,cz=0):
    return np.array([[s,0,0,0],[0,s,0,0],[0,0,s,0],[cx*(1-s),cy*(1-s),cz*(1-s),1]])
def rotate_x(radian):
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([[1,0, 0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
def rotate_y(radian):
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[ 0,0,0,1]])
def rotate_z(radian):
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([[c,-s,0,0],[s,c,0,0],[0, 0,1,0],[0, 0,0,1]])
def rotate_vec(gx, gy, gz, x,y,z, radian):
    rotZ = np.arctan2(y, x)
    rotZ_matrix = rotate_z(rotZ)
    (x, y, z, _) = np.dot(np.array([x,y,z,1]), rotZ_matrix)
    rotY = np.arctan2(x, z)
    matrix = eye_matrix(dx=-gx, dy=-gy, dz=-gz)
    matrix = np.dot(matrix, rotZ_matrix)
    matrix = np.dot(matrix, rotate_y(rotY))
    matrix = np.dot(matrix, rotate_z(radian))
    matrix = np.dot(matrix, rotate_y(-rotY))
    matrix = np.dot(matrix, rotate_z(-rotZ))
    matrix = np.dot(matrix, eye_matrix(dx=cx, dy=cy, dz=cz))
    return matrix
# > [Main_class], This method will accept edges, faces, and/or nodes, 
# and has transformation methods for the selected nodes
class MainGrid:
    def __init__(self, nodes=None):
        self.nodes = np.zeros((0,4))
        self.edges = []
        self.faces = []
        if nodes:
            self.add_nodes(nodes)
    def add_nodes(self, node_array):
        ones_added = np.hstack((node_array, np.ones((len(node_array),1))))
        self.nodes = np.vstack((self.nodes, ones_added))
    def add_edges(self, edge_list):
        self.edges += [edge for edge in edge_list if edge not in self.edges]
    def add_faces(self, face_list, face_colour=(255,255,255)):
        for node_list in face_list:
            num_nodes = len(node_list)
            if all((node < len(self.nodes) for node in node_list)):
                self.faces.append((node_list, np.array(face_colour, np.uint8)))
                self.add_edges([(node_list[n-1], node_list[n]) for n in range(num_nodes)])
    def output(self):
        if len(self.nodes) > 1:
            self.get_nodes()
        if self.edges:
            self.output_edges()
        if self.faces:
            self.output_faces()  
    def get_nodes(self):
        for i, (x,y,z,_) in enumerate(self.nodes):
            print("Node{}:({},{},{})".format(i, x, y, z))
    def output_edges(self):
        for i, (node1, node2) in enumerate(self.edges):
            print("Edge{}:{}->{}".format(i, node1, node2)) 
    def output_faces(self):
        for i, nodes in enumerate(self.faces):
            print("Face{}:({})".format(i, ", ".join(['{}'.format(n for n in nodes)])))
    def transform(self, transformation_matrix):
        self.nodes = np.dot(self.nodes, transformation_matrix)
    def find_centre(self):
        min_values = self.nodes[:,:-1].min(axis=0)
        max_values = self.nodes[:,:-1].max(axis=0)
        return 0.5*(min_values + max_values)
    def sorted_faces(self):
        return sorted(self.faces, key=lambda face: min(self.nodes[f][2] for f in face[0]))
    def update(self):
        pass
# > [Method_Grouping_Class] This method will group allready defined Grids() together into clusters
class MainGridClustering:
    def __init__(self):
        self.grids_dict = {}
    def add_grid(self, name, MainGrid):
        self.grids_dict[name] = MainGrid
    def output(self):
        for name, MainGrid in self.grids_dict.items():
            print (name)
            MainGrid.output()    
    def get_nodes(self):
        for name, MainGrid in self.grids_dict.items():
            print (name)
            MainGrid.get_nodes()
    def output_edges(self):
        for name, MainGrid in self.grids_dict.items():
            print (name)
            MainGrid.output_edges()
    def find_centre(self):
        min_values = np.array([MainGrid.nodes[:,:-1].min(axis=0) for MainGrid in self.grids_dict.values()]).min(axis=0)
        max_values = np.array([MainGrid.nodes[:,:-1].max(axis=0) for MainGrid in self.grids_dict.values()]).max(axis=0)
        return 0.5*(min_values + max_values)
    def transform(self, matrix):
        for MainGrid in self.grids_dict.values():
            MainGrid.transform(matrix)
    def update(self):
        for MainGrid in self.grids_dict.values():
            MainGrid.update()
# > [shaping_objects()] Declaring functions enabling the shape modelling and colloring of our desired objects or Grids
def shape_box(x,y,z,w,h,d):
    gen_box = MainGrid()
    gen_box.add_nodes(np.array([[nx,ny,nz] for nx in (x,x+w) for ny in (y,y+h) for nz in (z,z+d)]))
    gen_box.add_faces([(0,1,3,2), (7,5,4,6), (4,5,1,0), (2,3,7,6), (0,2,6,4), (5,7,3,1)])
    return gen_box
def shape_ball(x,y,z,rx,ry,rz,resolution=10):
    gen_ball = MainGrid()
    latitudes  = [n*np.pi/resolution for n in range(1,resolution)]
    longitudes = [n*2*np.pi/resolution for n in range(resolution)]
    gen_ball.add_nodes([(x + rx*np.sin(n)*np.sin(m), y - ry*np.cos(m), z - rz*np.cos(n)*np.sin(m)) for m in latitudes for n in longitudes])
    num_nodes = resolution*(resolution-1)
    gen_ball.add_faces([(m+n, (m+resolution)%num_nodes+n, (m+resolution)%resolution**2+(n+1)%resolution, m+(n+1)%resolution) for n in range(resolution) for m in range(0,num_nodes-resolution,resolution)])
    gen_ball.add_nodes([(x, y+ry, z),(x, y-ry, z)])
    gen_ball.add_faces([(n, (n+1)%resolution, num_nodes+1) for n in range(resolution)])
    start_node = num_nodes-resolution
    gen_ball.add_faces([(num_nodes, start_node+(n+1)%resolution, start_node+n) for n in range(resolution)])
    return gen_ball
def shape_plane(x,y,z,dx,dz,nx,nz):
    grid = MainGrid()
    grid.add_nodes([[x+n1*dx, y, z+n2*dz] for n1 in range(nx+1) for n2 in range(nz+1)])
    grid.add_edges([(n1*(nz+1)+n2,n1*(nz+1)+n2+1) for n1 in range(nx+1) for n2 in range(nz)])
    grid.add_edges([(n1*(nz+1)+n2,(n1+1)*(nz+1)+n2) for n1 in range(nx) for n2 in range(nz+1)])
    return grid
def shape_fractal(origin=(0,0,0), dimensions=(400,400), iterations=4, height=40):
    def midpoint(nodes):
        m = 1.0/ len(nodes)
        x = m * sum(n[0] for n in nodes) 
        y = m * sum(n[1] for n in nodes) 
        z = m * sum(n[2] for n in nodes) 
        return [x,y,z]
    (x,y,z) = origin
    (dx,dz) = dimensions
    nodes = [[x, y, z], [x+dx, y, z], [x+dx, y, z+dz], [x, y, z+dz]]
    edges = [(0,1), (1,2), (2,3), (3,0)]
    size = 2
    for i in range(iterations):
        for (n1, n2) in edges:
            nodes.append(midpoint([nodes[n1], nodes[n2]]))
        squares = [(x+y*size, x+y*size+1, x+(y+1)*size+1, x+(y+1)*size) for y in range(size-1) for x in range(size-1)]
        for (n1,n2,n3,n4) in squares:
            nodes.append(midpoint([nodes[n1], nodes[n2], nodes[n3], nodes[n4]]))
        nodes.sort(key=lambda node: (node[2],node[0]))
        size = size*2-1
        edges = [(x+y*size, x+y*size+1) for y in range(size) for x in range(size-1)]
        edges.extend([(x+y*size, x+(y+1)*size) for x in range(size) for y in range(size-1)])
        scale = height/2**(i*0.8)
        for node in nodes:
            node[1] += (random.random()-0.5)*scale
    grid = MainGrid(nodes)
    grid.add_edges(edges)
    grid = shape_fractal(origin = (0,400,0), iterations=1)
    grid.output()
# > [PyGame_vars] Variable declaration for the handles, lights, and visualisation settings via the imported pygame package
rotation_var = np.pi/36
movement_var = 2
key_to_function = {
    pygame.K_LEFT:   (lambda x: x.transform(eye_matrix(dx=-movement_var))),
    pygame.K_RIGHT:  (lambda x: x.transform(eye_matrix(dx= movement_var))),
    pygame.K_UP:     (lambda x: x.transform(eye_matrix(dy=-movement_var))),
    pygame.K_DOWN:   (lambda x: x.transform(eye_matrix(dy= movement_var))),
    pygame.K_EQUALS: (lambda x: x.scale(1.25)),
    pygame.K_MINUS:  (lambda x: x.scale(0.8)),
    pygame.K_q:      (lambda x: x.rotate('x', rotation_var)),
    pygame.K_w:      (lambda x: x.rotate('x',-rotation_var)),
    pygame.K_a:      (lambda x: x.rotate('y', rotation_var)),
    pygame.K_s:      (lambda x: x.rotate('y',-rotation_var)),
    pygame.K_z:      (lambda x: x.rotate('z', rotation_var)),
    pygame.K_x:      (lambda x: x.rotate('z',-rotation_var))}
light_movement = {
    pygame.K_q:      (lambda x: x.transform(rotate_x(-rotation_var))),
    pygame.K_w:      (lambda x: x.transform(rotate_x( rotation_var))),
    pygame.K_a:      (lambda x: x.transform(rotate_y(-rotation_var))),
    pygame.K_s:      (lambda x: x.transform(rotate_y( rotation_var))),
    pygame.K_z:      (lambda x: x.transform(rotate_z(-rotation_var))),
    pygame.K_x:      (lambda x: x.transform(rotate_z( rotation_var)))}
# > [Final_GridDisplay_Method] that displays the declared Grid or GridCluster with PyGame
class GridDisplay(MainGridClustering):
    def __init__(self, width, height, name="MainGrid Viewer"):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(name)
        self.grids_dict = {}
        self.grid_colours = {}
        self.object_to_update = []
        self.displayNodes = False
        self.displayEdges = True
        self.displayFaces = True
        self.perspective = False
        self.eyeX = self.width/2
        self.eyeY = 100
        self.view_vector = np.array([0, 0, -1])
        self.light = MainGrid()
        self.light.add_nodes([[0, -1, 0]])
        self.min_light = 0.02
        self.max_light = 1.0
        self.light_range = self.max_light - self.min_light 
        self.background = (10,10,50)
        self.nodeColour = (250,250,250)
        self.nodeRadius = 4
        self.control = 0
    def add_grid(self, name, MainGrid):
        self.grids_dict[name] = MainGrid
        self.grid_colours[name] = (250,250,250)
    def add_grid_cluster(self, MainGrid_group):
        for name, MainGrid in MainGrid_group.grids_dict.items():
            self.add_grid(name, MainGrid)
    def scale(self, scale):
        scale_matrix = eye_scale(scale, self.width/2, self.height/2, 0)
        self.transform(scale_matrix)
    def rotate(self, axis, amount):
        (x, y, z) = self.find_centre()
        translation_matrix1 = eye_matrix(-x, -y, -z)
        translation_matrix2 = eye_matrix(x, y, z)
        if axis == 'x':
            rotation_matrix = rotate_x(amount)
        elif axis == 'y':
            rotation_matrix = rotate_y(amount)
        elif axis == 'z':
            rotation_matrix = rotate_z(amount)
        rotation_matrix = np.dot(np.dot(translation_matrix1, rotation_matrix), translation_matrix2)
        self.transform(rotation_matrix)
    def display(self):
        self.screen.fill(self.background)
        light = self.light.nodes[0][:3]
        spectral_highlight = self.light.nodes[0][:3] + self.view_vector
        spectral_highlight /= np.linalg.norm(spectral_highlight)
        for name, MainGrid in self.grids_dict.items():
            nodes = MainGrid.nodes
            if self.displayFaces:
                for (face, colour) in MainGrid.sorted_faces():
                    v1 = (nodes[face[1]] - nodes[face[0]])[:3]
                    v2 = (nodes[face[2]] - nodes[face[0]])[:3]
                    normal = np.cross(v1, v2)
                    towards_us = np.dot(normal, self.view_vector)
                    if towards_us > 0:
                        normal /= np.linalg.norm(normal)
                        theta = np.dot(normal, light)
                        c = 0
                        if theta < 0:
                            shade = self.min_light *  colour
                        else:
                            shade = (theta * self.light_range + self.min_light) *  colour
                        pygame.draw.polygon(self.screen, shade, [(nodes[node][0], nodes[node][1]) for node in face], 0)
                if self.displayEdges:
                    for (n1, n2) in MainGrid.edges:
                        if self.perspective:
                            if MainGrid.nodes[n1][2] > -self.perspective and nodes[n2][2] > -self.perspective:
                                z1 = self.perspective/ (self.perspective + nodes[n1][2])
                                x1 = self.width/2  + z1*(nodes[n1][0] - self.width/2)
                                y1 = self.height/2 + z1*(nodes[n1][1] - self.height/2)                    
                                z2 = self.perspective/ (self.perspective + nodes[n2][2])
                                x2 = self.width/2  + z2*(nodes[n2][0] - self.width/2)
                                y2 = self.height/2 + z2*(nodes[n2][1] - self.height/2)
                                pygame.draw.aaline(self.screen,  (255,255,255), (x1, y1), (x2, y2), 1)
                        else:
                            pygame.draw.aaline(self.screen,  (255,255,255), (nodes[n1][0], nodes[n1][1]), (nodes[n2][0], nodes[n2][1]), 1)
            if self.displayNodes:
                for node in nodes:
                    pygame.draw.circle(self.screen, (255,255,255), (int(node[0]), int(node[1])), self.nodeRadius, 0)
        pygame.display.flip()
    def key_event(self, key):
        if key in key_to_function:
            key_to_function[key](self)
    def run(self):
        running = True
        key_down = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key_down = event.key
                elif event.type == pygame.KEYUP:
                    key_down = None
            if key_down:
                self.key_event(key_down)
            self.display()
            self.update()
        pygame.quit()
# Hello dear folks!
# Looking for a fully modular, open source, Pygame 3d-Engine? Well this may be a good start.
# GridMod is capable of visualising self made three dimensional shapes,
# by groupping nodes, vectors and matrices, and by applying common matrix operations 
# to those vertices we get to display those predefined 3d objects, 
# along with a scaling, rotation, and colour factor.
#That is it for now, yet more is coming. Feel free to let me know if you want to ask anything,
#   thereby suggessting what could further be done to make this even more awesome, 
#   or maybe informing us on what allready has been done around similar, simple open source python3D engines.
#   You are definetly also welcome to simply tell me if you liked it!!
# Ruf.
