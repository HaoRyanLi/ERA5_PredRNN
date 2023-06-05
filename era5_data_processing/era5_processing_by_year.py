import os
import sys
from time import sleep
#from osgeo import gdal
import numpy as np
from glob import glob
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import pygrib
from collections import defaultdict

data_path = '/scratch/09012/haoli1/ERA5/'
dataset_path = '/scratch/09012/haoli1/ERA5/dataset/'
infile = sys.argv[1]
print(f"infile: {infile}")
start_year = int(sys.argv[2])
time_step = int(sys.argv[3])
curr_month = int(sys.argv[4])

in_step = time_step
out_step = time_step


if start_year == 2000:
    years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006']
elif start_year == 2007:
    years = ['2007', '2008', '2009', '2010', '2011', '2012', '2013']
elif start_year == 2014:
    years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020']
elif start_year == 2021:
    years = ['2021', '2022']

norm_dict = {'u_wind':[-20,20],
            'v_wind':[-20,20],
            'temp':[210,305],
            'sea_press':[98000,105000],
            'precip':[-7,-2],
            'vapor':[-1,2.]}

lat = np.linspace(-np.pi/2, np.pi/2, 721)
lat_int = np.linspace(-np.pi/2, np.pi/2, 720)
dx = np.pi/720
lat_diff = np.reshape(lat_int-lat[:-1], (-1,1)).astype('float32')

def int_721_to_720(lat_diff, variables):
    # var_int = (variables[:,1:] - variables[:,:-1])/dx*lat_diff[None,:,:] + variables[:,:-1]
    var_int = (variables[:,1:] + variables[:,:-1])*0.5
    return var_int

for curr_year in years:
    u_wind_arr = []
    v_wind_arr = []
    sea_press_arr = []
    precip_arr = []
    temp_2m_arr = []
    data_ds1 = pygrib.open(data_path+infile)
    started = 0
    for ii,g in enumerate(data_ds1):
        if curr_year == str(g.date)[:4]:
            started = 1
        # The first one to process must have time the same as curr_year.
        if started == 0:
            continue

        if curr_year != str(g.date)[:4] and str(int(curr_year)-1)+'1231' !=str(g.date):
        # if curr_year != str(g.date)[:4]:
            if int(curr_year) < int(str(g.date)[:4]):
                break
            else:
                continue
        
        if '10 metre U wind component' == str(g.name):
            u_wind_arr.append(g.values.astype('float32'))
        elif '10 metre V wind component' == str(g.name):
            v_wind_arr.append(g.values.astype('float32'))
        elif 'Mean sea level pressure' == str(g.name):
            sea_press_arr.append(g.values.astype('float32'))
        elif 'Total precipitation' == str(g.name):
            precip_arr.append(g.values.astype('float32'))
        elif '2 metre temperature' == str(g.name):
            temp_2m_arr.append(g.values.astype('float32'))
        else:
            continue

    print(f"curr_year: {curr_year}",np.mean(precip_arr[0]))

    stack_u_wind = np.stack(u_wind_arr, axis=0)/20.
    stack_u_wind = int_721_to_720(lat_diff, stack_u_wind)
    stack_v_wind = np.stack(v_wind_arr, axis=0)/20.
    stack_v_wind = int_721_to_720(lat_diff, stack_v_wind)
    wind_speed = np.sqrt((stack_u_wind)**2+(stack_v_wind)**2)
    print(f"wind_speed shape: {wind_speed.shape}")

    stack_press = np.stack(sea_press_arr, axis=0)
    stack_press = int_721_to_720(lat_diff, stack_press)
    stack_press = (stack_press - norm_dict['sea_press'][0])/(norm_dict['sea_press'][1] - norm_dict['sea_press'][0])
    print(f"stack_press shape: {stack_press.shape}")

    stack_precip = np.stack(precip_arr, axis=0)
    stack_precip = int_721_to_720(lat_diff, stack_precip)
    if (stack_precip<0).any():
        print(f"Negative precip: {np.min(stack_precip[stack_precip<0])}")
    stack_precip[stack_precip<=0] = 1e-9
    stack_precip = (np.log10(stack_precip) - norm_dict['precip'][0])/(norm_dict['precip'][1] - norm_dict['precip'][0])
    print(f"stack_precip shape: {stack_precip.shape}")

    stack_temp = np.stack(temp_2m_arr, axis=0)
    stack_temp = int_721_to_720(lat_diff, stack_temp)
    stack_temp = (stack_temp - norm_dict['temp'][0])/(norm_dict['temp'][1] - norm_dict['temp'][0])

    final_data = np.stack([wind_speed, stack_press, stack_precip], axis=1)
    
    # final_data = final_data[:,:,:720,:]

    n_step = final_data.shape[0]
    print(f"final_data.shape: {final_data.shape}")
    print(f"curr_year: {curr_year}, Nan values num: {np.sum(np.isnan(final_data))}")

    final_clips = np.ones((2,int(n_step//in_step),2))*in_step
    clips = np.arange(0,n_step+1,out_step)
    final_clips[0,:,0] = clips[:-1]
    final_clips[1,:,0] = clips[1:]
    final_clips[1,:,1] = out_step
    print(final_clips.shape)

    final_ds = {}
    final_ds['input_raw_data'] = final_data.astype(np.float32)
    final_ds['dims'] = np.array([[3,720,1440]]).astype(np.int32)
    final_ds['clips'] = final_clips.astype(np.int32)

    np.savez(dataset_path+'era5_train_'+str(curr_month).zfill(2)+'01'+str(curr_year)+'_3_24hr.npz', **final_ds)