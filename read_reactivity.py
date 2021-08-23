#generate list of all reactivity data for jgi sequences from three sources
#The three sources are turnover data for SorbC, TropB, and AzaH, stereodata for ligands 1-2 and 18-2, and screening data for other ligands

#Also break up data into 3 sets
#Set 1: Training data for 3d convolutional network (1-2, 2-2, 17-2, 18-2, 32-2)
#Set 2: Test data for 3d convolutional network (a1, a2)
#Set 3: Test data with TropB, AzaH, and SorbC 
  #(1-2, 2-2, 3-2, 6-2, 7-2, 8-2, 9-2, 10-2, 11-2, 12-2, 13-2, 14-2, 15-2, 16-2, 17-2, 18-2, 20?, 22?, 26-2, 27-2, 28-2, 29-2, 30-2, 31-2, 32-2, 33-2)


import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

#Sources:
stereodata_file = 'qsar/reactivity_data/1st_JGI_library_stereodata.xlsx'
jgidata_file = 'qsar/reactivity_data/FMO_substrate_smiles_v2.xlsx'

# data for tropb, azah, sorbc: I will assume turnover > 100 is reactive
turnover_data = [ 
    [1000,296,0],
    [1000,0,0],
    [1000,0,0],
    [1000,1000,0],
    [1000,534,0],
    [976,78,0],
    [934,0,0],
    [787,0,479],
    [829,0,0],
    [669,0,331],
    [646,542,0],
    [377,385,371],
    [432,261,0],
    [793,0,0],
    [100,0,0],
    [402,0,0],
    [0,725,0],
    [0,639,0],
    [0,678,0],
    [0,632,0],
    [0,284,0],
    [0,16,0],
    [0,49,0],
    [0,13,0],
    [0,28,0],
    [0,16,68],
    [0,329,383],
    [0,277,646],
    [0,93,816],
    [0,308,656],
    [0,841,858],
    [0,418,889],
    [0,639,919],
]

#TEST DATA:
#Because there are two sources for test reactivity data, I'll read both and update any hits with set
# dictionary: model -> set(reactive), set(all_ligands)
train_dict = dict()
test_dict = dict()

#first go through stereodata and initialize dictionary to be two sets

#replace abbreivations with full name and remove Anc
def replace_abbrev(name):
  if name == name: #skip NaN
    name = name.lower().replace('anc', '').replace('.', '')
    if 'hypo' in name:
      name = name.split('_')[0]+'_hypothetical'
    elif 'func' in name:
      name = name.split('_')[0]+'_function_known'
  return name
  
#read in protein names, stereochemistry data for 18-2 and 1-2
protein_names = pd.read_excel(stereodata_file, sheet_name = 0, header = 6, nrows = 8, index_col = 0)
protein_names = protein_names.applymap(replace_abbrev)
stereo_data_18 = pd.read_excel(stereodata_file, sheet_name = 0, header = 26, nrows = 8, index_col = 0)
stereo_data_3mo = pd.read_excel(stereodata_file, sheet_name = 1, header = 16, nrows = 8, index_col = 0)
  
#go through all proteins
for column in protein_names.columns:
  for row in protein_names.index:
    protein = protein_names[column][row]
    if protein == protein:
      #stereodata
      protein = str(protein)
      data_18 = stereo_data_18[column][row]
      data_1 = stereo_data_3mo[column][row]
      
      #intialize dictionary
      train_dict[protein] = [set(), set()]
      train_dict[protein][1].add('18-2')
      train_dict[protein][1].add('1-2')
      
      if data_18 != '-':
        train_dict[protein][0].add('18-2')
      
      if data_1 != '-':
        train_dict[protein][0].add('1-2')
        


    
#next go through reactivity screening with other ligands
#jgi + other ligands
jgi_df = pd.read_excel(jgidata_file, sheet_name = 1, index_col = 0, header=6)  #read actual reactivity
jgi_df.drop(jgi_df.filter(regex="Unname"),axis=1, inplace=True) #drop unnamed columns, right now unnamed columns are those not part of original 33
for ligand in jgi_df.columns:
  if not ligand in ['a1', 'a2']: #screen out test ligands
    for model in jgi_df.index:
      reactivity = jgi_df[ligand][model]
      model = str(model)
      ligand = str(ligand)
      
      if not model in train_dict:
        train_dict[model] = [set(), set()]
      
      train_dict[model][1].add(ligand)
      if reactivity == '+':
        train_dict[model][0].add(ligand)
  else: #for test ligands
    for model in jgi_df.index:
      reactivity = jgi_df[ligand][model]
      model = str(model)
      ligand = str(ligand)
      
      if not model in test_dict:
        test_dict[model] = [set(), set()]
      
      test_dict[model][1].add(ligand)
      if reactivity == '+':
        test_dict[model][0].add(ligand)
        
        
        
for protein in train_dict:
  print(protein)
  print(train_dict[protein])
  
for protein in test_dict:
  print(protein)
  print(test_dict[protein])
  
#write model ligand pairs to txt files for train and test sets
  
with open('qsar/train_reactive_pairs.txt', 'w') as train_reactive_writer:
  with open('qsar/train_unreactive_pairs.txt', 'w') as train_unreactive_writer:
    with open('qsar/test_reactive_pairs.txt', 'w') as test_reactive_writer:
      with open('qsar/test_unreactive_pairs.txt', 'w') as test_unreactive_writer:
      
        models = list(train_dict.keys())
        for model in models:
          train_reactive_ligands = train_dict[model][0]
          train_unreactive_ligands = train_dict[model][1].difference(train_dict[model][0])
          if train_reactive_ligands:
            train_reactive_writer.write(model + ' ' + ' '.join(train_reactive_ligands) + '\n')
          if train_unreactive_ligands:
            train_unreactive_writer.write(model + ' ' + ' '.join(train_unreactive_ligands) + '\n')
            
            
        
        models = list(test_dict.keys())
        for model in models:
          test_reactive_ligands = test_dict[model][0]
          test_unreactive_ligands = test_dict[model][1].difference(test_dict[model][0])
          
          if test_reactive_ligands:
            test_reactive_writer.write(model + ' ' + ' '.join(test_reactive_ligands) + '\n')
          if test_unreactive_ligands:
            test_unreactive_writer.write(model + ' ' + ' '.join(test_unreactive_ligands) + '\n')



