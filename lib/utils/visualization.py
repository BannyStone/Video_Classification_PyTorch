import os
import matplotlib.pyplot as plt

log_file = "/home/leizhou/CVPR2019/vid_cls/output/kinetics400_mnet2_2D_length1_stride2_dropout0.2/log/logfile_25_Sep_2018_12:58:33"

class log_parser():
	def __init__(self, landmark, log_file, key_words=[], 
				 base_key_words=['Loss', 'Prec@1', 'Prec@5']):
		super(log_parser, self).__init__()
		with open(log_file) as f:
			self.lines = f.readlines()
		self.log_info = dict()
		self.landmark = landmark
		self.key_words = key_words + base_key_words
		for word in self.key_words:
			self.log_info[word] = []

	def __add__(self, other):
		"""Add two log parsers of same type.
		"""
		pass

	def parse(self):
		for line in self.lines:
			items = line.split()
			for word in self.key_words:
				if self.landmark not in items:
					break
				if word not in items:
					break
				ind = items.index(word) + 1
				try:
					self.log_info[word].append(float(items[ind]))
				except:
					self.log_info[word].append(items[ind])

	def convert_epoch_string(self):
		epochs = self.log_info['Epoch:']
		for idx, epoch_str in enumerate(epochs):
			epoch_num, fraction = epoch_str[1:-2].split("][")
			epoch = float(epoch_num) + eval(fraction)
			epochs[idx] = epoch

if __name__ == "__main__":
	# os.listdir()
	# Train Loss Parser
	tr_parser = log_parser("lr:", log_file, key_words=['Epoch:'])
	tr_parser.parse()
	tr_parser.convert_epoch_string()
	# Train Loss x, y axis
	x = tr_parser.log_info['Epoch:']
	loss = tr_parser.log_info['Loss']
	fig, ax = plt.subplots()
	ax.plot(x, loss, label='Loss')
	ax.set(xlabel="Epoch", ylabel='Loss', title='Train Loss')
	ax.grid()
	ax.legend(loc='upper right', shadow=False, fontsize='x-large')
	plt.show()

	# Test Acc Parser
	ts_parser = log_parser("Testing", log_file, key_words=['Epoch'])
	ts_parser.parse()
	# Test Acc x, y axis
	x = ts_parser.log_info['Epoch']
	top1 = ts_parser.log_info['Prec@1']
	top5 = ts_parser.log_info['Prec@5']
	fig, ax = plt.subplots()
	ax.plot(x, top1, label='Prec@1')
	ax.plot(x, top5, 'g--', label='Prec@5')
	ax.set(xlabel="Epoch", ylabel='Prec', title='Test Acc')
	ax.grid()
	ax.legend(loc='upper right', shadow=False, fontsize='x-large')
	plt.show()