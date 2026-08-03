import sys 
import os
import numpy as np
from matplotlib import pyplot as plt
 

import glob



wt_name='bar_urc'


verts_full_file= f'model_fidelity_study/{wt_name}/beam/beam.verts.all'
abscissa_full_file = f'model_fidelity_study/{wt_name}/beam'
beam_out_dir_name= f'{wt_name}/beam/npl_2_ar_2p5'

beam_in_dir_name = beam_out_dir_name       




def getInterpPtVal(pt,fieldVals,fieldPts,numPts):
    dim = len(fieldVals[0])
    nearVals = 1.0e+100*np.ones((numPts,(1+dim)),dtype=float)
    for i, fP in enumerate(fieldPts):
        vec = fP - pt
        dist = np.linalg.norm(vec)
        j = 0
        inserted = False
        while(j < numPts and (not inserted)):
            if(dist < nearVals[j,0]):
                for k in range((numPts-1),(j+1),-1):
                    nearVals[k] = nearVals[k-1]
                nearVals[j,0] = dist
                nearVals[j,1:(dim+1)] = fieldVals[i]
                inserted = True
            j = j + 1
    dsm = np.sum(nearVals[:,0])
    coef = 1.0/(nearVals[:,0] + 1.0e-8*dsm)
    sm = np.sum(coef)
    coef = (1.0/sm)*coef
    return list(np.matmul(coef,nearVals[:,1:(dim+1)]))
def get_path_interpolated_results(target_point_coords,vabs_results,node_coords,var_type):
    
    results = {}
    if len(vabs_results[0]) == 3:  
        keys=['1','2','3']
    elif len(vabs_results[0]) == 6:
        keys=['11','12','13','22','23','33']
    else:
        raise ValueError(f'Unknown error. Somethings wrong.')

    
    temp_result=[]
    for target_point_coord in target_point_coords:
        pt = np.pad(target_point_coord, (0, 1), 'constant', constant_values=(0, 0))
        temp_result.append(getInterpPtVal(pt,vabs_results,node_coords,4))

    temp_result = np.array(temp_result)
    for i_key,key in enumerate(keys):
        results[f'{var_type}_{key}']=list(temp_result[:,i_key])     
    return results       

def get_path_results(set_verts,vabs_results, var_type):

    results = {}
    if len(vabs_results[0]) == 3:  
        keys=['1','2','3']
    elif len(vabs_results[0]) == 6:
        keys=['11','12','13','22','23','33']
    else:
        raise ValueError(f'Unknown error. Somethings wrong.')


    for i_key,key in enumerate(keys):
        temp_result=[]
        for i_qp in set_verts:
            temp_result.append(vabs_results[i_qp][i_key])
        results[f'{var_type}_{key}']=temp_result

    return results



set_verts = {}
with open(f'{verts_full_file}', 'r') as f:
    lines = f.readlines()
    set_verts = {l.split()[0]:[int(i) for i in l.split()[1:]] for l in lines[:]}



#Get available stations
available_stations=[]
for file_path in glob.glob(beam_in_dir_name+'/'+wt_name+'*.in.S'):
    available_stations.append(int(file_path.split('-')[-3]))
available_stations.sort()




results = {}
#available_stations=[3]
for i_station,station in enumerate(available_stations):
    station_verts={}
    for path_name in set_verts.keys():
        #path_name='spanwise_aft_web_hp_adhesive'
        if 'span' in path_name:
            #print(path_name,i_station)
            if set_verts[path_name][i_station]:
                station_verts[path_name] = [set_verts[path_name][i_station]]
        else:
            this_station = int(path_name.split('_')[-1].strip('0'))
            if this_station == station:
                station_verts[path_name] = set_verts[path_name]
    print(f'Station: {i_station}, {station_verts}')


    file_path = f'{beam_in_dir_name}/{wt_name}-{station}-t-0.in.SM'
    i_start =2
    with open(f'{file_path}', 'r') as f:
        stresses = [[float(num) for num in line.split()[i_start:]] for line in f]
    with open(f'{file_path}', 'r') as f:
        qp_coords = np.array([[float(num) for num in line.split()[:2]] for line in f])

    file_path = f'{beam_in_dir_name}/{wt_name}-{station}-t-0.in.EM'
    i_start =2
    with open(f'{file_path}', 'r') as f:
        strains = [[float(num) for num in line.split()[i_start:]] for line in f]

    file_path = f'{beam_in_dir_name}/{wt_name}-{station}-t-0.in.U'
    i_start =3
    with open(f'{file_path}', 'r') as f:
        displacements = np.array([[float(num) for num in line.split()[i_start:]] for line in f])
    with open(f'{file_path}', 'r') as f:
        node_coords = np.array([[float(num) for num in line.split()[1:3]+['0.0']] for line in f])


    def plot_scatter(qps, qp_coords,vals):
        x=qp_coords[qps,0]
        y=qp_coords[qps,1]
        directions=['11','12','13','22','23','33']

        for i in range(6):
            z = vals[qps, i]/1e6
            plt.figure()
            plt.title(f's_{directions[i]}')
            plt.scatter(x, y, c=z, cmap='viridis') # s is a size of marker 
            plt.axis('equal')
            plt.colorbar().ax.set_ylabel('MPa', rotation=270, fontsize = 12, labelpad=18)


    
    #plot_scatter(list(range(0,len(qp_coords))),qp_coords,np.array(stresses))

    for path_name in station_verts.keys():
        #path_name = 'thickness_s2_003'
        this_set_qps =np.array(station_verts[path_name]) -1
        this_set_results = get_path_results(this_set_qps,stresses,'s')
        this_set_results.update(get_path_results(this_set_qps,strains,'e'))
        
        target_point_coords = qp_coords[this_set_qps]
        this_set_results.update(get_path_interpolated_results(target_point_coords,displacements,node_coords,'u'))

        # if len(this_set_qps) ==1 and path_name in set_coords.keys():
        #     set_coords[path_name].append(this_set_coords[0])
        #     set_block_names[path_name].append(this_set_block_names[0])

        # else:
        if path_name not in results.keys():
            results[path_name] = this_set_results
        else:
            for key in results[path_name].keys():
                results[path_name][key]+=this_set_results[key]




with open(f'{abscissa_full_file}/solid.abscissa', 'r') as f:
    lines = f.readlines()
    set_abscissas={l.split()[0]:[float(i) for i in l.split()[1:]] for l in lines[:]}


for path_name in results.keys():
    out_file_name = f'beam.{path_name}.values'
    out_file = open(f'{beam_out_dir_name}/{out_file_name}', 'w')

    out_file.write(f'non_dim_path {" ".join(map(str,set_abscissas[path_name]))}\n')
    
    for key in results[path_name].keys():
        out_file.write(f'{key} {" ".join(map(str,results[path_name][key]))}\n')
    out_file.close()
print()