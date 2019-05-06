import os
import csv
import collections
from collections import OrderedDict

frame_root = "/media/SSD/zhoulei/20bn-something-something-v1"

f_tr = open("sthv1_train_list.txt", 'w')
f_va = open("sthv1_val_list.txt", 'w')

name_id = OrderedDict()
with open('something-something-v1-labels.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=';')
	for i, row in enumerate(reader):
		assert(len(row) == 1), "the length of row must be one"
		name_id[row[0]] = i

with open('something-something-v1-train.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=';')
	for row in reader:
		dir_name = row[0]
		class_name = row[1]
		class_id = name_id[class_name]

		vid_dir = os.path.join(frame_root, dir_name)
		frame_num = len(os.listdir(vid_dir))

		line = ' '.join(("RGB/"+dir_name, str(frame_num), str(class_id)+'\n'))
		f_tr.write(line)

with open('something-something-v1-validation.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=';')
	for row in reader:
		dir_name = row[0]
		class_name = row[1]
		class_id = name_id[class_name]

		vid_dir = os.path.join(frame_root, dir_name)
		frame_num = len(os.listdir(vid_dir))

		line = ' '.join(("RGB/"+dir_name, str(frame_num), str(class_id)+'\n'))
		f_va.write(line)