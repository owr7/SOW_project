import random
import arlpy.uwapm as pm
import math
import numpy as np
from os import path
import sys
import scipy.io

def read_input_main(name_of_file):
    bathymetry=[]
    res=0
    ssp=[]
    surface_type=0
    ground_type=[]
    nodes_pos=[]
    schedule_matrix=[]
    freq = 25000
    delta_t = 0
    T = 0
    f = open(name_of_file, 'r')
    limits = []

    # Read the bathymetry
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            res, limits, bathymetry = read_bathymetry(line.split()[0])

    # Read the Sound speed profile
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            ssp = read_ssp(line.split()[0])

    # Read the surface type
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            surface_type = line.split()[0]

    # Read the ground type
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            ground_type = [float(k) for i, k in enumerate(line.split()) if i < 3]
    # Read nodes locations
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            nodes_pos = create_nodes_pos_array(int(line.split()[0]), line.split()[1], limits)

    # Read Schedule matrix
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            schedule_matrix = read_schedule_matrix(line.split()[0])

        # Read freq
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            freq = int(line.split()[0])

    # Read Time resolution
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            delta_t = int(line.split()[0])

    # Read Total Time
    for line in f:
        if line.find('>') >= 0:
            break
        if line.find('!') >= 0:
            T = int(line.split()[0])
    f.close()
    return bathymetry, limits, res, ssp, surface_type, ground_type, nodes_pos, schedule_matrix, freq, delta_t, T


def read_schedule_matrix(file_name):
    if not path.exists(file_name):
        print('The schedule matrix file:', file_name, 'does not exists in folder')
        sys.exit(0)

    f = open(file_name, 'r')
    m = []
    for line in f:
        m.append([int(k) for k in line.split()])
    schedule_matrix = np.zeros((m.__len__(), m[0].__len__()))
    for i in range(m.__len__()):
        for j in range(m[0].__len__()):
            schedule_matrix[i][j] = m[i][j]
    f.close()
    return schedule_matrix


# get position of node by the index on map.
def create_nodes_pos_array(random, first_pos_nodes, limits):
    nodes_pos = []
    if random < 0:
        f = open(first_pos_nodes, 'r')
        for i in f:
            direction = []
            for k in i.split():
                direction.append(int(k))
            nodes_pos.append(direction)
        f.close()
        return nodes_pos
    else:
        for i in range(random):
            nodes_pos.append([int(np.random.rand()*limits[0]+np.random.rand()*limits[1]),
                             int(np.random.rand()*limits[2]+np.random.rand()*limits[3]), int(np.random.rand() * 10 +
                             np.random.rand() * 40)])
        return nodes_pos



def read_ssp(file_name):
    if not path.exists(file_name):
        print('The ssp file:', file_name, 'does not exists in folder')
        sys.exit(0)

    f = open(file_name, 'r')
    ssp = []
    lines = f.readlines()
    for i in lines:
        if int(i.split()[0]) == 0:
            ssp.append([int(i.split()[0]), int(i.split()[1])])
        else:
            ssp.append([int(i.split()[0])+0.001, int(i.split()[1])])
    f.close()
    return ssp


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

    x_min = line_1[0]
    x_max = line_1[1]
    y_min = line_1[2]
    y_max = line_1[3]
    bathy = np.zeros((rowN, colN))
    k = 2
    for i in range(rowN):
        for J, j in enumerate(lines[k].split()):
            bathy[i][J] = int(j)
    limits = [int(x_min), int(x_max), int(y_min), int(y_max)]
    f.close()
    return res, limits, bathy


def find_closest_in_array(a, x: float):
    l = 0
    r = a.__len__()-1
    mid = int((r + l) / 2)
    while l < r:
        mid = int((r + l) / 2)
        if a[mid] == x or (mid == a.__len__() - 1 and x > a[mid]) or (mid == 0 and x < a[mid]):
            return mid
        if a[mid] < x:
            if abs(a[mid] - x) < abs(a[mid + 1] - x):
                return mid
            else:
                l = mid+1
                continue
        if a[mid] > x:
            if abs(a[mid] - x) < abs(a[mid - 1] - x):
                return mid
            else:
                r = mid-1
                continue
    mid = int((r + l) / 2)
    return mid


# limits = [min_x, max_x, min_y, max_y]
def drift(nodes_pos: list, matrix_x, matrix_y, dt: float, res: int, limits: list, bathy, step):
    #print('Before:', nodes_pos)
    for i, node in enumerate(nodes_pos):
        if node[0] < limits[0] or node[0] > limits[1] or node[1] < limits[2] or node[1] > limits[3]:
            # If one of the nodes goes out of map, then send signal to stop the simulation
            return nodes_pos, [-1, i]
        else:
            x = int(np.round((node[0] - limits[0]) / res))
            y = int(np.round((node[1] - limits[2]) / res))

            z_layers = [1, 2, 4, 8, 16, 32, 48, 64, 100, 128]
            z = find_closest_in_array(z_layers, node[2])
            if step != 0:

                vel_x = matrix_x[x][y][z]
                vel_y = matrix_y[x][y][z]

                node[0] += round(vel_x * dt, 3)
                node[1] += round(vel_y * dt, 3)
                # z doesn't change
            if node[0] < limits[0] or node[0] > limits[1] or node[1] < limits[2] or node[1] > limits[3]:
                return nodes_pos, [-1, i]
            # The new position
            x = int(np.round((node[0] - limits[0]) / res))
            y = int(np.round((node[1] - limits[2]) / res))

            if node[2] < 0:  # surface of water
                node[2] = 0

            a_x = (node[0] % res) / res
            a_y = (node[1] % res) / res
            x_t_down = int(np.floor((node[0] - limits[0]) / res))
            x_t_up = int(np.ceil((node[0] - limits[0]) / res))
            y_t_down = int(np.floor((node[1] - limits[2]) / res))
            y_t_up = int(np.ceil((node[1] - limits[2]) / res))
            
            bottom_depth = (bathy[y_t_down][x_t_down] * a_x * a_y + (1 - a_x) * a_y * bathy[y_t_down][x_t_up]
            + bathy[y_t_up][x_t_down] * a_x * (1 - a_y) + (1 - a_x) * (1 - a_y) * bathy[y_t_up][x_t_up])

            if node[2] > bottom_depth:
                node[2] = bottom_depth - 1
    return nodes_pos, [0, 0]


def key_func(a: list):
    return a[0]


# bathy is a matrix of 2 dim. in bathy[y][x]=h (z)
def create_spcifier_bathy(x_t: int, y_t: int, x_r: int, y_r: int, bathy, res: int, limits: list):
    new_bathy = []
    if x_r != x_t:
        m = abs((y_r - y_t) / (x_r - x_t))
    else:
        m = np.inf

    x_t_down = int(np.floor((x_t - limits[0]) / res))
    x_t_up = int(np.ceil((x_t - limits[0]) / res))
    y_t_down = int(np.floor((y_t - limits[2]) / res))
    y_t_up = int(np.ceil((y_t - limits[2]) / res))

    x_r_down = int(np.floor((x_r - limits[0]) / res))
    x_r_up = int(np.ceil((x_r - limits[0]) / res))
    y_r_down = int(np.floor((y_r - limits[2]) / res))
    y_r_up = int(np.ceil((y_r - limits[2]) / res))

    # coordinate x_t % res
    a_x = (x_t % res) / res
    a_y = (y_t % res) / res
    start_point_depth = (
            bathy[y_t_down][x_t_down] * a_x * a_y + (1 - a_x) * a_y * bathy[y_t_down][x_t_up]
            + bathy[y_t_up][x_t_down] * a_x * (1 - a_y) + (1 - a_x) * (1 - a_y) * bathy[y_t_up][x_t_up])


    #start_point_depth -= 0.0001
    # add the first point at the source
    new_bathy.append([0, start_point_depth])

    a_x = (x_r % res) / res
    a_y = (y_r % res) / res
    end_point_depth = (
            bathy[y_r_down][x_r_down] * a_x * a_y + (1 - a_x) * a_y * bathy[y_r_down][x_r_up]
            + bathy[y_r_up][x_r_down] * a_x * (1-a_y) + (1 - a_x) * (1 - a_y) * bathy[y_r_up][x_r_up])

    # add the last point at the receiver
    #end_point_depth -= 0.0001
    new_bathy.append([np.sqrt((x_r-x_t)**2+(y_r-y_t)**2), end_point_depth])

    x_curr = 0
    y_curr = 0
    x_direction = 0
    y_direction = 0

    if y_t < y_r:
        y_direction = 1
    elif y_t > y_r:
        y_direction = -1


    if x_t < x_r:
        x_curr = x_t_up
        x_direction = 1
        _curr = y_t + y_direction * m * (res - x_t % res)
    elif x_t > x_r:
        x_curr = x_t_down
        x_direction = -1
        y_curr = y_t + y_direction * m * (x_t % res)


    steps = int(np.round(abs(x_r - x_t) / res))
    for i in range(steps):
        x_curr = int(x_curr)
        up_y = int(math.ceil((y_curr - limits[2]) / res))
        down_y = int(np.floor((y_curr - limits[2]) / res))
        a_y = (y_curr % res) / res
        if up_y > int(limits[3]/res) or down_y < int(limits[2]/res):
            break
        h = np.round(bathy[up_y][x_curr] * a_y + bathy[down_y][x_curr] * (1 - a_y))
        #h -= 0.0001
        dist_from_source = math.sqrt((x_curr * res - limits[1] - x_t) ** 2 + (y_curr - y_t) ** 2)
        new_bathy.append([dist_from_source, h])
        # calculate the next point
        x_curr += x_direction
        y_curr += m * y_direction * res

    if y_t < y_r:
        y_curr = y_t_up
        y_direction = 1
        x_curr = x_t + x_direction * 1/m * (res - y_t % res)
    elif y_t > y_r:
        y_curr = y_t_down
        y_direction = -1
        x_curr = x_t + x_direction * 1 / m * (y_t % res)


    steps = int(np.round(abs(y_r - y_t) / res))
    for i in range(steps):
        y_curr = int(y_curr)
        up_x = int(math.ceil((x_curr - limits[0]) / res))
        down_x = int(np.floor((x_curr - limits[0]) / res))
        a_x = (x_curr % res) / res
        if up_x > int(limits[1]/res) or down_x < int(limits[0]/res):
            break
        h = np.round(bathy[y_curr][up_x] * a_x + bathy[y_curr][down_x] * (1 - a_x))
        #h -= 0.0001
        new_bathy.append([math.sqrt((x_curr - x_t) ** 2 + (y_curr * res -limits[3] - y_t) ** 2), h])
        # calculate the next point
        y_curr += y_direction
        x_curr += 1 / m * x_direction * res

    # sort the points according to the distance from the source
    new_bathy.sort(key=key_func)
    new_bathy = [[round(k[0], 2), k[1]] for k in new_bathy]

    j = 0
    n = new_bathy.__len__()
    while j < n:
        if j > 0:

            if new_bathy[j][0] <= new_bathy[j-1][0]:# and new_bathy[j][1] == new_bathy[j-1][1]:
                new_bathy.remove(new_bathy[j-1])
                n -= 1
                continue
        j += 1

    return new_bathy

def fix_ssp(ssp, max_depth):
    #print(ssp)
    count = 0
    for l in ssp:
        count += 1
        if l[0] > max_depth:
            break
    if count > 3:
        return ssp
    else:
        if count == 2:
            add_point_1 = ssp[1][0]/3
            add_point_2 = 2 * add_point_1
            add_ss_1 = ssp[0][1] * 2 / 3 + ssp[1][1] * 1 / 3
            add_ss_2 = ssp[0][1] * 1 / 3 + ssp[1][1] * 2 / 3
            return [ssp[0], [add_point_1, add_ss_1], [add_point_2, add_ss_2], ssp[1]]
        elif count == 3:
            add_point_1 = ssp[1][0]/2
            add_ss_1 = ssp[0][1] / 2 + ssp[1][1] / 2
            return [ssp[0], [add_point_1, add_ss_1], ssp[1], ssp[2]]





def call_bellhop(bathy, ssp, res, x_t: int, y_t: int, z_t: int, x_r: int, y_r: int, z_r: int,
                 ground_type, surface_type, limits, freq):

    new_bathy = create_spcifier_bathy(x_t, y_t, x_r, y_r, bathy, res, limits)
    if z_t == 0:
        z_t = 0.1
    if z_r == 0:
        z_r = 0.1
    if z_r > new_bathy[len(new_bathy)-1][1]:
        z_r = new_bathy[len(new_bathy)-1][1] - 1
    if z_t > new_bathy[0][1]:
        z_t = new_bathy[len(new_bathy)-1][1] - 1

    max_depth_for_new_bathy = max([k[1] for k in new_bathy])
    ssp = fix_ssp(ssp, max_depth_for_new_bathy)
    #print(ssp)
    ##round(math.sqrt((x_r - x_t) ** 2 + (y_r - y_t) ** 2) - 0.0001, 2),
    env = pm.create_env2d(depth=new_bathy,
                          soundspeed=ssp,
                          rx_range=new_bathy[len(new_bathy)-1][0],
                         
                          rx_depth=z_r, tx_depth=z_t,
                          bottom_absorption=ground_type[0],
                          bottom_density=ground_type[1],
                          bottom_roughness=ground_type[2])


    env['rx_range'] = new_bathy[new_bathy.__len__()-1][0] - 0.001
    if surface_type == 'waves':
        env['surface'] = np.array([[r, 0.5+0.5*np.sin(2*np.pi*0.005*r)] for r in np.linspace(0, env['rx_range']+1
                                                                                             , 1001)])
    flg = False
    try:
        arrivals = pm.compute_arrivals(env)
    except Exception:
        print('******The trassmission not arrived******')
        flg = True
        pass
    if flg:
        return 'not arrived', -1
    ir = pm.arrivals_to_impulse_response(arrivals, fs=int(freq))
    return ir, 1


if __name__ == '__main__':
    pm.models()

    bathy, limits, res, ssp, surface_type, ground_type, nodes_pos, schedule_matrix, freq, delta_t, time_duration = \
        read_input_main('INPUT/bellhop_input_file')
    vel_matrix_x = np.load('SHY_OUTPUT/matrix_vel_x.dat', allow_pickle=True)
    vel_matrix_y = np.load('SHY_OUTPUT/matrix_vel_y.dat', allow_pickle=True)
    
    output_ir_matrix = open('OUTPUT/ir_output.csv', 'w')
    ir = []
    output_location_matrix = open('OUTPUT/location_output.csv', 'w')
    print('Time, Transmiter, Receiver, Lenght', file=output_ir_matrix)
    print('Time, Node, x, y, z', file=output_location_matrix)
    num_of_steps = math.ceil(time_duration / delta_t)
    print('nodes_pos:', nodes_pos)

    for i in range(num_of_steps):
        print("Step number:", i, "of simulation")
        receivers = [r for r, k in enumerate(schedule_matrix[:, i]) if k == -1]
        nodes_pos, flg = drift(nodes_pos, vel_matrix_x, vel_matrix_y, delta_t, res, limits, bathy, i)
        if flg[0] < 0:
            print("The simulation stopped  because node", flg[1], "goes out from map")
            sys.exit(0)
        for k, node in enumerate(nodes_pos):
            print(i * delta_t, ',', k, ',', round(node[0], 3), ',', round(node[1], 3), ',', round(node[2], 3),
                  file=output_location_matrix)
            if schedule_matrix[k][i] > 0:
                for r_node in receivers:
                    response, check = call_bellhop(bathy, ssp, res, round(node[0], 3), round(node[1], 3), round(node[2], 3),
                                                round(nodes_pos[r_node][0], 3),
                                                round(nodes_pos[r_node][1], 3),
                                                round(nodes_pos[r_node][2], 3), ground_type, surface_type, limits, freq)
                    print((i+1) * delta_t, ', ', k, ', ', r_node, ', ', end='', file=output_ir_matrix)
                    ir.append([(i+1)*delta_t, k, r_node, response])
                    if check < 0:
                        print(response,end=',', file=output_ir_matrix)
                    else:
                        for r in response:
                            print(r, end=',', file=output_ir_matrix)
                    print('\n', file=output_ir_matrix)
    output_location_matrix.close()
    output_ir_matrix.close()
    scipy.io.savemat('OUTPUT/ir.mat', mdict={'arr': ir})
