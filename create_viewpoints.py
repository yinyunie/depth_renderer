# This script is to create 20 viewpoints (vertices of a regular dodecahedron) around shapes.
import math

if __name__ == '__main__':
    phi = (1 + math.sqrt(5)) / 2. # golden_ratio
    circumradius = math.sqrt(3)
    distance = circumradius*1.2
    dodecahedron = [[-1, -1, -1],
                    [ 1, -1, -1],
                    [ 1,  1, -1],
                    [-1,  1, -1],
                    [-1, -1,  1],
                    [ 1, -1,  1],
                    [ 1,  1,  1],
                    [-1,  1,  1],
                    [0, -phi, -1 / phi],
                    [0, -phi,  1 / phi],
                    [0,  phi, -1 / phi],
                    [0,  phi,  1 / phi],
                    [-1 / phi, 0, -phi],
                    [-1 / phi, 0,  phi],
                    [ 1 / phi, 0, -phi],
                    [ 1 / phi, 0,  phi],
                    [-phi, -1 / phi, 0],
                    [-phi,  1 / phi, 0],
                    [ phi, -1 / phi, 0],
                    [ phi, 1 / phi, 0]]

    # get Azimuth, Elevation angles
    # Azimuth varies from -pi to pi
    # Elevation from -pi/2 to pi/2
    view_points = open('./view_points.txt', 'w+')
    for vertice in dodecahedron:
        elevation = math.asin(vertice[2] / circumradius)
        azimuth = math.atan2(vertice[1], vertice[0])
        view_points.write('%f %f %f %f\n' % (azimuth, elevation, 0., distance))
    view_points.close()


