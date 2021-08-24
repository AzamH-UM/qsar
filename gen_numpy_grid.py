#generate grids for every model ligand pair in test and training set
import os
import sys
import numpy as np


#grid params
cmds = []   #commands to use gen_grid.inp
pairs = []  #hold model lig pairs
griddirs = [] #hold path to model lig grids
dimension = 20  #xyz dimensions of grid
spacing = .5    #spacing between grid potints
max_vdw = 1000   
max_elec = 1000
min_vdw = -1000
min_elec = -1000
elem = int(dimension / spacing + 1)
already_generated = 0

#function to submit protein and ligand to gen_grid.inp
def make_grid_cmd(model, ligand, griddir, dimension):
  charmm_exec = '/home/brookscl/charmm/c44dev/install_domdec-omm-fftdock/bin/charmm'
  inp = 'qsar/gen_grid.inp'
  cmd = f'{charmm_exec} -i {inp} protein={model} ligand={ligand} griddir={griddir} dim={dimension} spacing={spacing}'
  model_lig = f'{model}_{ligand}'
  return cmd, model_lig
  
#function to read model ligand pairs
#dataset: "train" or "test"
#reactivity: "reactive" or "unreactive"
def read_model_lig_list(dataset, reactivity):
  global already_generated
  with open(f'qsar/{dataset}_{reactivity}_pairs.txt', 'r') as pair_file:
    for line in pair_file.read().splitlines():
      model, ligands = line.split()[0], line.split()[1:]
      griddir = f'qsar/{dataset}_{reactivity}_grids'
      for ligand in ligands:
        if os.path.isfile(os.path.join(griddir, f'{model}_{ligand}_gridpoints.log')):
          already_generated += 1
        cmd, pair = make_grid_cmd(model, ligand, griddir, dimension)
        cmds.append(cmd)
        pairs.append(pair)
        griddirs.append(griddir)
      
#read charmm grid to numpy array
def read_log(logfile):
  vdw_array = np.zeros((elem, elem, elem), dtype = float)
  elec_array = np.zeros_like(vdw_array)
  x_count = elem - 1
  y_count = elem - 1
  z_count = elem - 1
  counter = 0
  
  with open(logfile, 'r') as logfile_reader:
    lines = logfile_reader.read().splitlines()
  
  for x in range(x_count, -1, -1):
    for y in range(y_count, -1, -1):
      for z in range(z_count, -1, -1):
        #print(x , y , z )
        line = lines[counter]
        grid_x, grid_y, grid_z, vdw, elec = line.split()
        vdw = float(vdw)
        elec = float(elec)
        
        #because elec can have both large positive and negative values
        elec = abs(elec)
       
        
        '''
        if vdw > max_vdw:
          vdw = max_vdw
        if elec > max_elec:
          elec = max_elec
          
        if vdw < min_vdw:
          vdw = min_vdw
        if elec < min_elec:
          elec = min_elec
          
        '''
        
        #add data to numpy array: in numpy first index is depth, second is row, third is column
        #I want z to select matrix, x to be column, and y to be row
        vdw_array[z][y][x] = vdw
        elec_array[z][y][x] = elec
        
        counter+= 1
  
  #normalize data (min max feature scaling) x' = (x - min(x)) / (max(x) - min(x))
  #new: add smallest value + 1 and take log
  print('unchanged arrays')
  print(vdw_array)
  print(elec_array)
  
  if np.amin(vdw_array) < 0:
    vdw_array = vdw_array - np.amin(vdw_array)
    
  if np.amin(elec_array) < 0:
    elec_array = elec_array - np.amin(elec_array)
  
  vdw_array = vdw_array + 1
  elec_array = elec_array + 1
  
  print('min + 1')
  print(vdw_array)
  print(elec_array)
  
  vdw_array = np.log10(vdw_array)
  elec_array = np.log10(elec_array)
  
  print('log10')
  print(vdw_array)
  print(elec_array)
  
  norm_vdw_array = (vdw_array - np.amin(vdw_array)) / float(np.amax(vdw_array) - np.amin(vdw_array))
  norm_elec_array = (elec_array - np.amin(elec_array)) / float(np.amax(elec_array) - np.amin(elec_array))
  
  print('normalized')
  print(norm_vdw_array)
  print(norm_elec_array)
  
  grid_array = np.stack([norm_vdw_array, norm_elec_array], axis = 3)
  
  
  assert counter == elem ** 3
  print(f'read logfile: {logfile}')
  
  return grid_array

read_model_lig_list('train','reactive')
read_model_lig_list('train','unreactive')
read_model_lig_list('test','reactive')
read_model_lig_list('test','unreactive')


#submit grid job based on slurm id
if len(sys.argv) < 2:
  print(len(cmds), 'total grids')
  print(already_generated, 'grids generated')
  
else:
  #run charmm
  i_cmd = int(sys.argv[1])
  cmd = cmds[i_cmd]
  pair = pairs[i_cmd]
  griddir = griddirs[i_cmd]
  
  print(cmd)
  print(pair)
  print(griddir)
  
  out = os.popen(cmd).read()
  #print(out)
  
  #process charmm output to numpy
  logfile = os.path.join(griddir, pair + '_gridpoints.log')
  asciifile = os.path.join(griddir, pair + '.ascii')
  grid_array = read_log(logfile)
  #save to .npy file
  grid_file = os.path.join(griddir, pair + '.npy')
  np.save(grid_file, grid_array)
  
  #delete old grid files
  os.remove(logfile)
  os.remove(asciifile)
  
  
  





