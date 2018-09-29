import os
import matplotlib.pyplot as plt

files = os.listdir("log")
for ind, file in enumerate(files):
	if not file.startswith('logfile'):
		files.pop(ind)
print(files)
files = ["log/"+file for file in files]
files.sort()

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
		assert hasattr(self, "hist") and hasattr(other, "hist"), "Parse before adding."
		for key in self.hist.keys():
			assert key in other.hist, "Mush share key when adding."
			self.hist[key].update(other.hist[key])
		return self

	def parse(self):
		# parse info into list
		for line in self.lines:
			items = line.strip().split()
			if self.landmark not in items:
				continue
			for word in self.key_words:
				assert(word in items), "Key word should be in target line."
			for word in self.key_words:
				ind = items.index(word) + 1
				if word == "Epoch:":
					self.log_info[word].append(items[ind])
				else:
					self.log_info[word].append(float(items[ind]))

		# convert epoch string
		self.convert_epoch_string()
		# find the key for the later dict
		if "Epoch:" in self.key_words:
			key = "Epoch:"
		else:
			key = "Epoch"

		# build hist
		self.hist = {}
		for word in self.key_words:
			if "Epoch" not in word:
				self.hist[word] = {}
				for k, v in zip(self.log_info[key], self.log_info[word]):
					self.hist[word].update({k: v})

	def convert_epoch_string(self):
		if "Epoch:" in self.log_info:
			epochs = self.log_info['Epoch:']
			for idx, epoch_str in enumerate(epochs):
				epoch_num, fraction = epoch_str[1:-2].split("][")
				epoch = float(epoch_num) + eval(fraction)
				epochs[idx] = epoch

def plot(files, tr_landmark="lr:", ts_landmark="Testing"):
	if not isinstance(files, list):
		files = [files]

	file = files[0]
	tr_parser_base = log_parser(tr_landmark, files[0], key_words=['Epoch:'])
	tr_parser_base.parse()
	ts_parser_base = log_parser(ts_landmark, files[0], key_words=['Epoch'])
	ts_parser_base.parse()
	if len(files) > 1:
		for file in files[1:]:
			tr_parser = log_parser(tr_landmark, file, key_words=['Epoch:'])
			tr_parser.parse()
			ts_parser = log_parser(ts_landmark, file, key_words=['Epoch'])
			ts_parser.parse()
			tr_parser_base += tr_parser
			ts_parser_base += ts_parser

	fig, ax = plt.subplots()
	ax.plot(tr_parser_base.hist['Loss'].keys(), tr_parser_base.hist['Loss'].values(), label='Train Loss')
	ax.plot(ts_parser_base.hist['Loss'].keys(), ts_parser_base.hist['Loss'].values(), label='Val Loss')
	ax.set(xlabel="Epoch", ylabel='Loss', title='Loss')
	ax.grid()
	ax.legend(loc='upper right', shadow=False, fontsize='x-large')
	plt.show()

	fig, ax = plt.subplots()
	ax.plot(ts_parser_base.hist['Prec@1'].keys(), ts_parser_base.hist['Prec@1'].values(), label='Prec@1')
	ax.plot(ts_parser_base.hist['Prec@5'].keys(), ts_parser_base.hist['Prec@5'].values(), 'g--', label='Prec@5')
	ax.set(xlabel="Epoch", ylabel='Prec', title='Test Acc')
	ax.grid()
	ax.legend(loc='lower right', shadow=False, fontsize='x-large')
	plt.show()

if __name__ == "__main__":
	plot(files)
