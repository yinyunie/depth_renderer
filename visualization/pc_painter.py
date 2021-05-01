'''
This script includes Classes to draw 3D point cloud from depth maps

Author: ynie
Date: Jan 2020
'''
import os
from tools.read_and_write import read_exr, read_txt, read_image
from PIL import Image
from copy import copy
import numpy as np
import vtk
from tools.utils import dist_to_dep
import point_cloud_utils as pcu
import random

class PC_from_DEP(object):
    def __init__(self, metadata_dir, camera_path, view_ids, with_normal=True):
        dist_map_dir = [os.path.join(metadata_dir, 'depth_{0:03d}.exr'.format(view_id)) for view_id in view_ids]
        cam_RT_dir = [os.path.join(camera_path, 'cam_RT','cam_RT_{0:03d}.txt'.format(view_id)) for view_id in view_ids]
        rgb_img_dir = [os.path.join(metadata_dir, 'color_{0:03d}.png'.format(view_id)) for view_id in view_ids]

        dist_maps = read_exr(dist_map_dir)
        self._cam_K = np.loadtxt(os.path.join(camera_path, 'cam_K/cam_K.txt'))
        self._cam_RTs = read_txt(cam_RT_dir)
        self._rgb_imgs = read_image(rgb_img_dir)
        self._with_normal = with_normal

        self._depth_maps = np.float32(dist_to_dep(dist_maps, [self._cam_K]*len(view_ids)))
        self._point_clouds = self.get_point_cloud(self.depth_maps, [self._cam_K]*len(view_ids), self._cam_RTs, self._rgb_imgs)
        if with_normal:
            self._point_clouds['normal'] = self.get_point_normal(self._point_clouds)

    @property
    def depth_maps(self):
        return self._depth_maps

    @property
    def cam_K(self):
        return self._cam_K

    @property
    def cam_RTs(self):
        return self._cam_RTs

    @property
    def rgb_imgs(self):
        return self._rgb_imgs

    @property
    def point_clouds(self):
        return self._point_clouds

    @property
    def with_normal(self):
        return self._with_normal

    @staticmethod
    def get_point_normal(point_clouds):
        pc_counts = [pc.shape[0] for pc in point_clouds['pc']]
        pc_all = np.vstack(point_clouds['pc'])
        normal_all = pcu.estimate_point_cloud_normals(pc_all, k=16)

        view_id = 0

        corrected_normals = []
        for cam, pc in zip(point_clouds['cam'], point_clouds['pc']):
            n_pc = normal_all[sum(pc_counts[:view_id]):sum(pc_counts[:view_id + 1])]
            to_inverse_flags = np.matmul((pc - cam['pos'])[:, np.newaxis, :], n_pc[:,:,np.newaxis]).flatten()>0
            n_pc[to_inverse_flags,:] *= -1
            corrected_normals.append(n_pc)
            view_id += 1

        return corrected_normals

    def set_camera(self, position, focal_point, cam_K):
        camera = vtk.vtkCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point[0])
        camera.SetViewUp(*focal_point[1])
        camera.SetViewAngle((2*np.arctan(cam_K[1][2]/cam_K[0][0]))/np.pi*180)
        return camera

    def get_point_cloud(self, depth_maps, cam_Ks, cam_RTs, rgb_imgs=None):
        '''
        get point cloud from depth maps
        :param depth_maps: depth map list
        :param cam_Ks: corresponding camera intrinsics
        :param cam_RTs: corresponding camera rotations and translations
        :param rgb_imgs: corresponding rgb images
        :return: aligned point clouds in the canonical system with color intensities.
        '''
        point_list_canonical = []
        camera_positions = []
        color_intensities = []

        if not isinstance(rgb_imgs, np.ndarray):
            rgb_imgs = 32*np.ones([depth_maps.shape[0], depth_maps.shape[1], depth_maps.shape[2], 3], dtype=np.uint8)

        for depth_map, rgb_img, cam_K, cam_RT in zip(depth_maps, rgb_imgs, cam_Ks, cam_RTs):
            u, v = np.meshgrid(range(depth_map.shape[1]), range(depth_map.shape[0]))
            u = u.reshape([1, -1])[0]
            v = v.reshape([1, -1])[0]

            z = depth_map[v, u]

            # remove infinitive pixels
            non_inf_indices = np.argwhere(z < np.inf).T[0]

            color_indices = rgb_img[v, u][non_inf_indices]
            z = z[non_inf_indices]
            u = u[non_inf_indices]
            v = v[non_inf_indices]

            # calculate coordinates
            x = (u - cam_K[0][2]) * z / cam_K[0][0]
            y = (v - cam_K[1][2]) * z / cam_K[1][1]

            point_cam = np.vstack([x, y, z]).T

            point_canonical = (point_cam - cam_RT[:, -1]).dot(cam_RT[:,:-1])
            cam_pos = - cam_RT[:, -1].dot(cam_RT[:,:-1])
            focal_point = ([0, 0, 1] - cam_RT[:, -1]).dot(cam_RT[:,:-1])
            up = np.array([0,-1,0]).dot(cam_RT[:,:-1])

            cam_pos = {'pos':cam_pos, 'fp':focal_point, 'up':up}

            point_list_canonical.append(point_canonical)
            camera_positions.append(cam_pos)
            color_intensities.append(color_indices)

        return {'pc': point_list_canonical, 'cam': camera_positions, 'color': color_intensities}

    def draw_color(self, view = 0):
        if view != 'all':
            rgb_imgs = [copy(self.rgb_imgs[view])]
        else:
            rgb_imgs = copy(self.rgb_imgs)

        n_cols = min(5, rgb_imgs.shape[0])
        n_rows = np.ceil(rgb_imgs.shape[0]/n_cols).astype(np.uint8)
        len_row = self.rgb_imgs.shape[1]
        len_col = self.rgb_imgs.shape[2]

        image_template = np.zeros([n_rows * len_row, n_cols * len_col, 3], dtype=np.uint8)

        for view_id, rgb_img in enumerate(rgb_imgs):
            i = view_id // n_cols
            j = view_id % n_cols

            image_template[i * len_row: (i + 1) * len_row, j * len_col: (j + 1) * len_col] = rgb_img

        Image.fromarray(image_template).show()

    def draw_depth(self, view = 0):
        if view != 'all':
            depth_maps = [copy(self.depth_maps[view])]
        else:
            depth_maps = copy(self.depth_maps)

        n_cols = min(5, depth_maps.shape[0])
        n_rows = np.ceil(depth_maps.shape[0]/n_cols).astype(np.uint8)
        len_row = self.depth_maps.shape[1]
        len_col = self.depth_maps.shape[2]

        image_template = np.zeros([n_rows * len_row, n_cols * len_col], dtype=np.uint8)

        for view_id, depth_map in enumerate(depth_maps):
            i = view_id // n_cols
            j = view_id % n_cols

            z_max = depth_map[depth_map < np.Inf].max()
            z_min = depth_map[depth_map < np.Inf].min()

            upper_bound = 16
            lower_bound = 255

            depth_map = lower_bound + (upper_bound - lower_bound)/(z_max-z_min)*(depth_map-z_min)

            image_template[i * len_row: (i + 1) * len_row, j * len_col: (j + 1) * len_col] = np.uint8(depth_map)

        Image.fromarray(image_template).show()

    def set_mapper(self, prop, mode):

        mapper = vtk.vtkPolyDataMapper()

        if mode == 'model':
            mapper.SetInputConnection(prop.GetOutputPort())

        elif mode == 'box':
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(prop)
            else:
                mapper.SetInputData(prop)

            # mapper.SetScalarRange(0, 7)

        else:
            raise IOError('No Mapper mode found.')

        return mapper

    def set_actor(self, mapper):
        '''
        vtk general actor
        :param mapper: vtk shape mapper
        :return: vtk actor
        '''
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def set_points_property(self, point_cloud_dict):
        point_clouds = point_cloud_dict['pc']
        point_colors = point_cloud_dict['color']

        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        for point_cloud, point_color in zip(point_clouds, point_colors):
            x3 = point_cloud[:, 0]
            y3 = point_cloud[:, 1]
            z3 = point_cloud[:, 2]

            for x, y, z, c in zip(x3, y3, z3, point_color):
                id = points.InsertNextPoint([x, y, z])
                colors.InsertNextTuple3(*c)
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)
        point.GetPointData().SetActiveScalars('Color')

        return point

    def set_axes_actor(self):
        '''
        Set camera coordinate system
        '''
        transform = vtk.vtkTransform()
        transform.Translate(0., 0., 0.)
        # self defined
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        axes.SetTotalLength(0.3, 0.3, 0.3)

        axes.SetTipTypeToCone()
        axes.SetConeRadius(30e-2)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(40e-3)

        vtk_textproperty = vtk.vtkTextProperty()
        vtk_textproperty.SetFontSize(1)
        vtk_textproperty.SetBold(True)
        vtk_textproperty.SetItalic(False)
        vtk_textproperty.SetShadow(True)

        for label in [axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()]:
            label.SetCaptionTextProperty(vtk_textproperty)

        return axes

    def set_arrow_actor(self, startpoint, vector):
        '''
        Design an actor to draw an arrow from startpoint to startpoint + vector.
        :param startpoint: 3D point
        :param vector: 3D vector
        :return: an vtk arrow actor
        '''
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.4)
        arrow_source.SetTipRadius(0.2)
        arrow_source.SetShaftRadius(0.05)

        vector = vector/np.linalg.norm(vector)*0.04

        endpoint = startpoint + vector

        # compute a basis
        normalisedX = [0 for i in range(3)]
        normalisedY = [0 for i in range(3)]
        normalisedZ = [0 for i in range(3)]

        # the X axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(endpoint, startpoint, normalisedX)
        length = math.Norm(normalisedX)
        math.Normalize(normalisedX)

        # the Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalisedX, arbitrary, normalisedZ)
        math.Normalize(normalisedZ)


        # the Y axis is Z cross X
        math.Cross(normalisedZ, normalisedX, normalisedY)

        # create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalisedX[i])
            matrix.SetElement(i, 1, normalisedY[i])
            matrix.SetElement(i, 2, normalisedZ[i])

        # apply the transform
        transform = vtk.vtkTransform()
        transform.Translate(startpoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # create a mapper and an actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()

        mapper.SetInputConnection(arrow_source.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)

        return actor

    def set_normal_actor(self, pointcloud, color = (1, 0, 0), count = 1000):
        random_pc = np.vstack(pointcloud['pc'])
        random_normals = np.vstack(pointcloud['normal'])
        rand_ids = np.random.choice(random_pc.shape[0], count)

        random_pc = random_pc[rand_ids]
        random_normals = random_normals[rand_ids]

        arrow_actors = []
        for index in range(random_pc.shape[0]):
            arrow_actor = self.set_arrow_actor(random_pc[index], random_normals[index])
            arrow_actor.GetProperty().SetColor(color)
            arrow_actors.append(arrow_actor)

        return arrow_actors

    def set_render(self):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''set camera property'''
        cam_id = 7
        center = self._point_clouds['cam'][cam_id]['pos']
        focal_point = self._point_clouds['cam'][cam_id]['fp']
        up = self._point_clouds['cam'][cam_id]['up']
        camera = self.set_camera(center, [focal_point, up], self._cam_K)
        renderer.SetActiveCamera(camera)

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        '''draw point clouds'''
        point_actor = self.set_actor(self.set_mapper(self.set_points_property(self._point_clouds), 'box'))
        point_actor.GetProperty().SetPointSize(2)
        renderer.AddActor(point_actor)

        '''draw normals'''
        if self._with_normal:
            normal_actors = self.set_normal_actor(self._point_clouds)
            for normal_actor in normal_actors:
                renderer.AddActor(normal_actor)

        renderer.SetBackground(1., 1., 1.)

        return renderer

    def set_render_window(self):

        render_window = vtk.vtkRenderWindow()
        renderer = self.set_render()
        render_window.AddRenderer(renderer)

        if hasattr(self, 'depth_maps'):
            window_size = self.depth_maps[0].shape
        else:
            window_size = [512, 512]

        render_window.SetSize(window_size[1], window_size[0])

        return render_window

    def draw3D(self):
        '''
        Visualize 3D point clouds from multi-view depth maps
        :return:
        '''
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window = self.set_render_window()
        render_window_interactor.SetRenderWindow(render_window)
        render_window.Render()
        render_window_interactor.Start()



