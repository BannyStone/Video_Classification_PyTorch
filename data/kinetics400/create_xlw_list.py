access = "access/"
import os
from tqdm import tqdm

# with open("kinetics_val_list.txt") as f_old:
# 	with open("kinetics_val_list_xlw", 'w') as f_new:
# 		old_lines = f_old.readlines()
# 		for line in old_lines:
# 			vid_path, num_fr, label = line.strip().split()
# 			if os.path.exists(access+vid_path):
# 				new_num_fr = len(os.listdir(access+vid_path))
# 				f_new.write(" ".join([vid_path, str(new_num_fr), label]) + '\n')

# with open("kinetics_train_list.txt") as f_old:
# 	with open("kinetics_train_list_xlw", 'w') as f_new:
# 		old_lines = f_old.readlines()
# 		for line in old_lines:
# 			vid_path, num_fr, label = line.strip().split()
# 			if os.path.exists(access+vid_path):
# 				new_num_fr = len(os.listdir(access+vid_path))
# 				f_new.write(" ".join([vid_path, str(new_num_fr), label]) + '\n')

with open("kinetics_train_list_xlw") as f:
	lines = f.readlines()
	for line in tqdm(lines):
		vid_path, num_fr, label = line.strip().split()
		images = os.listdir(access+vid_path)
		images.sort()
		last_image = images[-1]
		# import pdb
		# pdb.set_trace()
		if int(last_image[6:-4]) != int(num_fr):
			print(vid_path)