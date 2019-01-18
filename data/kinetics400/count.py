frames = []
with open("kinetics_val_list.txt") as f:
	lines = f.readlines()
	for line in lines:
		items = line.strip().split()
		frames.append(int(items[1]))

total = len(frames)
count60 = 0
count120 = 0
count240 = 0

for fr in frames:
	if fr > 60:
		count60 += 1
	if fr > 120:
		count120 += 1
	if fr > 240:
		count240 += 1

print("60: ", count60, total, count60/total)
print("120: ", count120, total, count120/total)
print("240: ", count240, total, count240/total)