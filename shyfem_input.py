import matplotlib.pyplot as plt_
import numpy as np
from os import path
import sys


def read_bathymetry(file_name):
    if not path.exists(file_name):
        print('The bathmetry file:', file_name, 'does not exists in folder')
        sys.exit(0)

    f = open(file_name, 'r')
    lines = f.readlines()
    line_0 = lines[0].split()
    line_1 = lines[1].split()

    res = int(line_0[0])
    rowN = int(line_0[1])
    colN = int(line_0[2])
    bottom_depth = 0

    x_min = line_1[0]
    x_max = line_1[1]
    y_min = line_1[2]
    y_max = line_1[3]
    bathy = np.zeros((rowN, colN))
    k = 2
    for i in range(rowN):
        for J, j in enumerate(lines[k].split()):
            bathy[i][J] = int(j)
            if int(j) > bottom_depth:
                bottom_depth = int(j)
        k += 1
    limits = [int(x_min), int(x_max), int(y_min), int(y_max)]
    f.close()
    return res, rowN, colN, limits, bathy, bottom_depth


def matrix2list(m):
    m_list = []
    for row in m:
        for i in row:
            m_list.append(i)
    return m_list


def bathy2grd(x, y, h, filename):
    f = open(filename, 'w')
    x_list = matrix2list(x)
    y_list = matrix2list(y)
    h_list = matrix2list(h)
    for k in range(x_list.__len__()):
        print(x_list[k], y_list[k], h_list[k], file=f)
    f.close()


def coast2grd(contour, filename):
    f = open(filename, 'w')
    for k in contour:
        print(k[0], k[1], file=f)
    f.close()


if __name__ == '__main__':
    # sys.argv[1] = bathymetry file name
    res, row, col, limits, h, bottom_depth = read_bathymetry(sys.argv[1])
    # x_coast, y_coast = read_coast("INPUT/coast_file.txt")
    xi = np.linspace(limits[0], limits[1], row)
    yi = np.linspace(limits[2], limits[3], col)
    x, y = np.meshgrid(xi, yi)
    x = np.transpose(x)
    y = np.transpose(y)

    hS = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            hS[i][j] = h[i][j]

    hS[0, :] = [-1 for i in range(col)]
    hS[:, 0] = [-1 for i in range(row)]
    hS[row - 1, :] = [-1 for i in range(col)]
    hS[:, col - 1] = [-1 for i in range(row)]

    cont = plt_.contour(x, y, hS, [0, 1])
    dat0 = cont.allsegs[0][0]

    # TODO: get friction, wave and dlat
    coast2grd(dat0, 'mpcoast.dat')
    bathy2grd(x, y, h, 'mpbathy.dat')
