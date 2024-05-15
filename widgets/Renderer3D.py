import math
import tkinter
from tkinter import ttk, filedialog, colorchooser, messagebox

import sv_ttk


###Time Me
import time

def time_me(f):
    '''Decorator function to time functions' runtime in ms'''
    def wrapper(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        print(f'function: {f.__name__} took {(time.time()-start)*1000:.4f}ms')
        return res
    return wrapper


###extract data
import re

import numpy as np

def extract_data(file):
    """
    @brief: This function takes in a .obj file and returns a list containing
            the indexes of the vertexes that make up the face, and a dict
            that contains the index of the vertex, along with its coordinates

    @param: .obj file

    @ret  : verticies (the dict containing coordinates)
    @ret  : faces (the list containing the vertexes' indexes)
    """

    verticies = np.empty((1, 3))
    faces = []

    # Read more about how waveform (.obj) files are structured to understand
    # how this code exactly works, but shortly:
    #   * If the line starts with a "v", then that's a vertex and what follows is
    #     its X, Y, Z coordinates (foats)
    #   * If the line starts with a "f", then that's a face and what follows is
    #     the list of verticies to be connected to create a face
    #     (formatted a bit strangely though, I recommend checking an example)

    for line in file.readlines():
        if line[0:2] == "v ":
            pt = np.array([[float(x) for x in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)]])
            verticies = np.concatenate([verticies, pt])
        elif line[0:2] == "f ":
            faces.append([int(vertex.split("/")[0]) for vertex in line[2:-2].split(' ')])
    return verticies, faces

###MAIN

class Canvas3D(tkinter.Canvas):
    def __init__(self, frame, parent=None):
        if parent is None:
            parent=tkinter.Tk()
        self.parent=parent
        #inherit
        try:
            self.assets=self.parent.assets
        except:
            self.assets=Assets()
        try:
            self.data=self.parent.data
        except:
            self.data=Shared_Data()
        self.parent_frame=frame
        self._geometry_handler = Geometry(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        super().__init__(self.parent_frame)
        self.pack(expand=1, fill='both')
        self._canvas.bind("<Configure>", self.__resized)
        self._changed = False

        self._geometry_handler.reset_rotation()  
        self._file_name.set(file_path.split('/')[-1])
        with open(file_path) as file:
            self._geometry_handler.upload_object(*obj_files_handler.extract_data(file))
            self._file_exists = True


    def __get_canvas_shape(self):
        """returns the shape of the canvas holding the visualized frame"""
        self.update()
        return self._canvas.winfo_width(), self._canvas.winfo_height()

    def __resized(self, *args):
        '''Callback to the window resize events'''
        w, h = self.__get_canvas_shape()
        if self._canvas_w != w or self._canvas_h != h:
            # Keep the object in the middle of the canvas
            self._geometry_handler.update_position((w-self._canvas_w)//2, (h-self._canvas_h)//2)
            self._canvas_w = w
            self._canvas_h = h
            self.__changed()

    def __changed(self, *args):
        '''Signal to the rendering function that something has changed in the object'''
        self._changed = True


    def render(self):

        self.__set_rotations()
        # self.__set_zoom()

        #Delete all the previous points and lines in order to draw new ones
        self._canvas.delete("all")
        self.__update_colors()
        self.__draw_object()
        self._changed = False

    def __set_zoom(self):
        self._geometry_handler.set_zoom(self._zoom_slider.get())

    def __set_rotations(self):
        '''Set the required rotations for the geometry handler'''
        x, y, z = self._geometry_handler.orientation
        self._geometry_handler.reset_rotation(
            x=x, 
            y= y, 
            z=z
        )

    def __draw_point(self, point: 'tuple(int, int)') -> None:
        '''Draw a point on the canvas'''
        self._canvas.create_oval(point[0], point[1],
                                    point[0], point[1],
                                    width=self.POINT_SIZE,
                                    fill=self.POINT_COLOR)
    
    @time_me
    def __draw_faces(self, points: dict) -> None:
        ''''''
        for face in self._geometry_handler.faces:
            # Grab the points that make up that specific face
            to_draw = [points[f] for f in face]
            
            #for point in to_draw:
            #    if(point[0] < 0 or
            #       point[1] < 0 or
            #       point[0] > self.CANVAS_WIDTH or
            #       point[1] > self.CANVAS_HEIGHT
            #    ):
            #        continue # Don't draw points that are out of the screen
            #    # This is the slowest part of the GUI
            #    self.__draw_point(point)

            self._canvas.create_polygon(to_draw, outline=self._line_color_holder, fill=self._fill_color_holder)

    def __draw_object(self):
        '''Draw the object on the canvas'''
        projected_points = self._geometry_handler.transform_object()
        self.__draw_faces(projected_points)


####GEOMETRY FILE
import math
import numpy as np
import numba

@numba.njit(nogil=True, cache=True, fastmath=True)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    @brief: Matrix multiplication from scratch
    @param A: First matrix 
    @param B: Second matrix
    @Note: A, and B must be multiplicable matrices; that must have a common dimension
    @return C: Product of A and B
    '''
    rows, cols = A.shape[0], B.shape[1]
    C = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            for k in range(rows):
                C[i, j] += A[i, k] * B[k, j]
    return C

@numba.njit(nogil=True, cache=True)
def max_3d_array(arr: np.ndarray, axis: int) -> float:
    '''
    Find the maximum in a miltidimensional array, in the provided axis
    '''
    max_ = -np.inf
    for i in arr:
        if i[axis] >= max_: 
            max_ = i[axis] 
    return max_

@numba.njit(nogil=True, cache=True)
def min_3d_array(arr: np.ndarray, axis: int) -> float:
    '''
    Find the minimum in a miltidimensional array, in the provided axis
    '''
    min_ = np.inf
    for i in arr:
        if i[axis] <= min_: 
            min_ = i[axis] 
    return min_

class Geometry:
    '''
    Geometry handling class (linear algebra)
    '''
    OBJECT_SCALE = 2000 # Maybe make this dynamic depending on the object size
    
    def __init__(self, canvas_width: int, canvas_height: int) -> None:
        '''
        '''
        self._obj_position = np.array((canvas_width//2, canvas_height//2))
        self._zoom = 50.0
        self._angle_x = 0.0
        self._angle_y = 0.0
        self._angle_z = 0.0
        self._faces = None
        self._verticies = None

    def upload_object(self, verts: np.ndarray, faces: list) -> None:
        '''Uploads the verticies and faces to manipulate'''
        self._verticies = self.__normalize_3d_array(verts, axis=0)
        self._faces = faces

    def update_position(self, x: int, y: int) -> None:
        '''Update x, y position of the object'''
        self._obj_position[0] += x
        self._obj_position[1] += y
    
    @time_me
    def transform_object(self) -> 'list(list(int, int))':
        '''Retur the points of the object transformed according to the current pose'''
        rot_x, rot_y, rot_z = self.__calculate_rot_matrix()
        projected_points = []
        for pt in self._verticies:
            x, y = self.__transform_point(pt, rot_x, rot_y, rot_z, self._zoom, self._obj_position, self.OBJECT_SCALE)
            projected_points.append([x, y])
        return projected_points

    @property
    def faces(self) -> list:
        '''Get the faces formed between the points'''
        return self._faces

    @property
    def zoom(self) -> int:
        '''Get the current zoom value'''
        return self._zoom

    @property
    def orientation(self) -> 'tuple(float, float, float)':
        '''Returns the object's current angles'''
        return self._angle_x, self._angle_y, self._angle_z

    def set_zoom(self, zoom: float) -> None:
        '''Set the new zoom value'''
        self._zoom = zoom

    def step_rotation(self, 
                      x: float = 0.0,
                      y: float = 0.0,
                      z: float = 0.0
    ) -> None:
        '''Increment the orientation of the object on its axis'''
        self._angle_x += x
        self._angle_y += y
        self._angle_z += z

    def reset_rotation(self, 
                       x: float = None,
                       y: float = None,
                       z: float = None
    ) -> None:
        '''Reset the rotation to a specific position, if provided, else to 0'''
        self._angle_x = 0 if x is None else x
        self._angle_y = 0 if y is None else y
        self._angle_z = 0 if z is None else z

    @staticmethod
    @numba.njit(nogil=True, cache=True, fastmath=True)
    def __normalize_3d_array(arr: np.ndarray, 
                             range: 'tuple(float, float)' = (-1, 1),
                             axis: int = 2
    ) -> np.ndarray:
        '''
        @brief: Normalize an array values within a range based on a specified axis
        @param arr: The array to be normalized
        @param range: Normalized values range (min, max)
        @param axis: the axis to normalize based on
        @return arr: The normalized array
        '''
        mnx = min_3d_array(arr, 0)
        mxx = max_3d_array(arr, 0)
        mnz = min_3d_array(arr, 2)
        mxz = max_3d_array(arr, 2)
        mny = min_3d_array(arr, 1)
        mxy = max_3d_array(arr, 1)

        if axis == 0:
            diff = mxx - mnx
        elif axis == 1:
            diff = mxy - mny
        else:
            diff = mxz - mnz
        
        for pt in arr:
            pt[0] = (((pt[0]-mnx)*(range[1]-range[0]))/diff) + range[0]
            pt[1] = (((pt[1]-mny)*(range[1]-range[0]))/diff) + range[0]
            pt[2] = (((pt[2]-mnz)*(range[1]-range[0]))/diff) + range[0]    
        return arr

    @staticmethod
    @numba.njit(nogil=True, cache=True, fastmath=True)
    def __transform_point(point: np.ndarray, 
                          rotation_x: np.ndarray, 
                          rotation_y: np.ndarray, 
                          rotation_z: np.ndarray,
                          zoom: float,
                          obj_position: 'list[int, int]',
                          obj_scale: int
    ) -> 'tuple(int, int)':
        '''
        @brief: Rotate the point in 3axis according to the provided rotation matrices
        @param point: 3D point
        @param rotation_x: Rotation matrix on X axis
        @param rotation_y: Rotation matrix on Y axis
        @param rotation_z: Rotation matrix on Z axis
        @param zoom: Zoom value
        @param obj_position: Object position within the screen
        @param obj_scale: Object scale
        @return transformed point: 2D tranformed projection of the 3D point
        '''
        # Rotate point on the Y, X, and Z axis respectively
        rotated_2d = matmul(rotation_y, point.reshape((3, 1)))
        rotated_2d = matmul(rotation_x, rotated_2d)
        rotated_2d = matmul(rotation_z, rotated_2d)

        # Project 3D point on 2D plane
        z = 0.5 / (zoom - rotated_2d[2][0])
        projection_matrix = np.array(((z, 0, 0), (0, z, 0)))
        projected_2d = matmul(projection_matrix, rotated_2d)

        x = int(projected_2d[0][0]*obj_scale) + obj_position[0]
        # The (-) sign in the Y is because the canvas' Y axis starts from Top to Bottom, 
        # so without the (-) sign, our objects would be presented upside down
        y = -int(projected_2d[1][0]*obj_scale) + obj_position[1]

        return x, y

    def __calculate_rot_matrix(self) -> 'tuple(np.array, np.array, np.array)':
        '''
        Calculate the rotation matrices on X, Y, and Z axis 
        that correspond to the current requested rotation
        '''
        rotation_x = np.array(
            (
                (1,               0        ,               0         ),
                (0, math.cos(self._angle_x), -math.sin(self._angle_x)),
                (0, math.sin(self._angle_x),  math.cos(self._angle_x))
            )
        )

        rotation_y = np.array(
            (
                (math.cos(self._angle_y), 0, -math.sin(self._angle_y)),
                (            0          , 1,             0           ),
                (math.sin(self._angle_y), 0,  math.cos(self._angle_y))
            )
        )

        rotation_z = np.array(
            (
                (math.cos(self._angle_z), -math.sin(self._angle_z), 0),
                (math.sin(self._angle_z),  math.cos(self._angle_z), 0),
                 (           0           ,              0          , 1)
            )
        )
        return rotation_x, rotation_y, rotation_z

