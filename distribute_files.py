import os
import shutil

src = os.getcwd()
full_input_dir = os.path.join(src, 'all_inputs')
all_inputs = os.listdir(full_input_dir)
num_dir_split = 16
target_dir = os.path.join(src, 'distributed_files_1')
os.mkdir(target_dir)
input_dir_names = []
output_dir_names = []
for i in range(0,num_dir_split):
    input_dir_names.append('input_partition_' + str(i))
    output_dir_names.append('output_partition_' + str(i))
    new_input_dir = os.path.join(target_dir, input_dir_names[-1])
    new_output_dir = os.path.join(target_dir, output_dir_names[-1])
    os.mkdir(new_input_dir)
    os.mkdir(new_output_dir)
partition_index = 0
for file_name in all_inputs:
    full_file_name = os.path.join(full_input_dir, file_name)
    partition_input_dir = input_dir_names[partition_index % num_dir_split]
    partition_dir = os.path.join(target_dir, partition_input_dir)
    shutil.copy(full_file_name, partition_dir)
    partition_index += 1