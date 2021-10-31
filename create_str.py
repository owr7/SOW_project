import sys
import numpy as np


def generate_str_file(filename, wind_vel, wind_dir, dragco, waves, friction, salt, temp, dlat, wind_file='', nodes=None):
    str_file = open(filename, 'w')
    bathy_file = open('basbathy.grd', 'r')

    print('This file was generated by matlab for a random scenario\n', file=str_file)
    print('$title', file=str_file)
    print('\tRandomly generated scenario', file=str_file)
    print('\tscenario', file=str_file)
    print('\tbasbathy', file=str_file)
    print('$end\n', file=str_file)


    print('$para', file=str_file)
    print('\tdate = 0', file=str_file)
    print('\titanf = 0 itend =', time_sim,'idt = 1', file=str_file)
    print('\tidtout =', time_sim,'itmout =', time_sim, file=str_file)
    print('\tidtext =', time_sim, 'itmext =', time_sim, file=str_file)
    print('\tilin = 1', file=str_file)
    print('\tcoumax = 0.8  itsplt = 1', file=str_file)
    print('\tdiftur = 0.001 vistur = 0.001', file=str_file)  # צמיגות

    # roughnes
    print('\tireib = 6', file=str_file)
    print('\tczdef =', friction, file=str_file)

    print('\tisalt = 1 salref = ', salt, file=str_file)  # salinity constant 38.5 ppm
    print('\titemp = 1 temref = ', temp, file=str_file)  # temperature 16 degree Celsius
    print('\ticonz = 1 conref = 1', file=str_file)
    print('\tshpar = 0.2 thpar = 0.2 chpar = 0.2', file=str_file)
    print('\titvd = 2 itvdv = 1', file=str_file)

    print('\tilytyp = 3 hlvmin = 0.5', file=str_file)
    # print('\tampar = 0.60 azpar = 0.60', file=str_file)

    # latitude
    print('\ticor = 1 dlat = ', dlat, file=str_file)

    # type of wind input data: in x and y components, m/sec...
    if wind_vel == '0':
        print('\tiwtype = 0', file=str_file)
    else:
        print('\tiwtype = 3', file=str_file)

    print('\titdrag = 0', file=str_file)
    # drag coefficient for wind stress
    print('\tdragco = ', dragco,  file=str_file)

    print('\twsmax = 200', file=str_file)  # maximum wind speed (if error occured in the wind speed file
    print('\taapar = 0.0', file=str_file)

    print('$end', file=str_file)

    print('$levels', file=str_file)

    print('\t1. 2. 4. 8. 16. 32. 64. 128. 150. 180. 200.', file=str_file)
    print('$end\n', file=str_file)

    # waves
    print('$waves', file=str_file)
    print('\tiwave =', waves, file=str_file)  # wave={0,1}
    print('$end\n', file=str_file)

    # wind
    if wind_vel != '0':
        if wind_vel != '-1':
            f_wind = open('INPUT/wind_file', 'w')
            print('0', wind_vel, wind_dir, file=f_wind)
            a = int(wind_dir) + int(np.random.random()*30)
            print('25', wind_vel, a, file=f_wind)
            print('50', wind_vel, wind_dir, file=f_wind)
            a = int(wind_dir) + int(np.random.random()*30)
            print('75', wind_vel, a, file=f_wind)
            print('100', wind_vel, wind_dir, file=f_wind)
            f_wind.close()
            print('$name', file=str_file)
            print('\twind = "INPUT/wind_file"', file=str_file)
            print('$end\n', file=str_file)
        else:
            print('$name', file=str_file)
            print('\twind = ""', file=str_file)
            print('$end\n', file=str_file)

    str_file.close()


def prepare_bathy_coordinates():
    bathy_file = open('basbathy.grd', 'r')
    new_bathy = open('bathy_file', 'w')
    nodes_list = []
    for line in bathy_file.readlines():
        line_ = line.split()
        if line_.__len__() > 0:
            if int(line_[0]) == 1:
                print('\t', line_[1], line_[3], line_[4], file=new_bathy)
               #if abs(float(line_[3]) - 400) < 50:
                nodes_list.append(line_[1])
    bathy_file.close()
    new_bathy.close()
    return nodes_list


if __name__ == '__main__':
    wind_vel = sys.argv[1]
    wind_dir = sys.argv[2]
    dragco = sys.argv[3]
    waves = sys.argv[4]
    friction = sys.argv[5]
    salt = sys.argv[6]
    temp = sys.argv[7]
    dlat = sys.argv[8]
    wind_file = sys.argv[9]
    time_sim = sys.argv[10]

    if wind_dir == '-1':
        wind_dir = 0
    if waves == '-1':
        waves = 0
    if friction == '-1':
        friction = 0.1
    if salt == '-1':
        salt = 38.5
    if temp == '-1':
        temp = 16
    if dlat == '-1':
        dlat = 30
    if dragco == '-1':
        dragco = 2.5e-3

    flg = 0
    if wind_file != '0':
        flg = 1
    
    nodes_list = prepare_bathy_coordinates()
    if flg == 1:
        generate_str_file('scenario.str', wind_vel, wind_dir, dragco, waves, friction, salt, temp, dlat, wind_file, time_sim)
    else:
        generate_str_file('scenario.str', wind_vel, wind_dir, dragco, waves, friction, salt, temp, dlat, time_sim)
