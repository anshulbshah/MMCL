import os
import pathlib
import subprocess 

destination_folder = ''
source_folder = ''

f = open('classes_imagenet100.txt','r')
classes100 = f.read().splitlines() 

pathlib.Path(os.path.join(destination_folder,'CLS-LOC','train')).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(destination_folder,'CLS-LOC','val')).mkdir(parents=True, exist_ok=True)

for cl in classes100:
	cmd = f'ln -s {source_folder}/train/{cl} {destination_folder}/CLS-LOC/train/{cl}'

	subprocess.Popen(cmd, stdout = subprocess.PIPE,shell=True)

	cmd = f'ln -s {source_folder}/val/{cl} {destination_folder}/CLS-LOC/val/{cl}'
	subprocess.Popen(cmd, stdout = subprocess.PIPE,shell=True)


