import pdb

# extract target 200-class videos from the original videos
with open('kinetics_train_list.txt') as tr400:
	with open('Mini-Kinetics-200/train_ytid_list.txt') as miniTr:
		with open('kinetics200_train_list_org.txt', 'w') as tr200:
			# build indices for original 400-class train list
			lines = tr400.readlines()
			ytid_line_dict = dict()
			for line in lines:
				ytid = line.strip().split()[0].split('/')[1]
				ytid_line_dict[ytid] = line
			# extract target lines and write them into tr200 file
			lines = miniTr.readlines()
			for line in lines:
				ytid = line.strip()
				if ytid in ytid_line_dict:
					target_line = ytid_line_dict[ytid]
					tr200.write(target_line)
				else:
					print("{} is not in original video list".format(ytid))

with open('kinetics_val_list.txt') as va400:
	with open('Mini-Kinetics-200/val_ytid_list.txt') as miniVa:
		with open('kinetics200_val_list_org.txt', 'w') as va200:
			# build indices for original 400-class val list
			lines = va400.readlines()
			ytid_line_dict = dict()
			for line in lines:
				ytid = line.strip().split()[0].split('/')[1]
				ytid_line_dict[ytid] = line
			# extract target lines and write them into va200 file
			lines = miniVa.readlines()
			for line in lines:
				ytid = line.strip()
				if ytid in ytid_line_dict:
					target_line = ytid_line_dict[ytid]
					va200.write(target_line)
				else:
					print("{} is not in original video list".format(ytid))

# summarize all the 200 categories of Mini-Kinetics
# Train and val
cats_tr = set()
cats_va = set()

with open("kinetics200_train_list_org.txt") as f:
	lines = f.readlines()
	for line in lines:
		label_id = int(line.strip().split()[-1])
		cats_tr.add(label_id)

with open("kinetics200_val_list_org.txt") as f:
	lines = f.readlines()
	for line in lines:
		label_id = int(line.strip().split()[-1])
		cats_va.add(label_id)

assert(cats_tr == cats_va)

# build 400-class 200-class dictionary
_400_200_dict = dict()
for i, cat in enumerate(cats_tr):
	_400_200_dict[cat] = i

with open('400_200_label_mapping.txt', 'w') as f:
	for key, value in _400_200_dict.items():
		f.write("{} {}\n".format(key, value))

with open('kinetics200_train_list_org.txt') as f_src:
	with open('kinetics200_train_list.txt', 'w') as f_dst:
		lines = f_src.readlines()
		for line in lines:
			items = line.strip().split()
			items[-1] = str(_400_200_dict[int(items[-1])])
			new_line = ' '.join(items)
			f_dst.write(new_line + '\n')

with open('kinetics200_val_list_org.txt') as f_src:
	with open('kinetics200_val_list.txt', 'w') as f_dst:
		lines = f_src.readlines()
		for line in lines:
			items = line.strip().split()
			items[-1] = str(_400_200_dict[int(items[-1])])
			new_line = ' '.join(items)
			f_dst.write(new_line + '\n')

# pdb.set_trace()