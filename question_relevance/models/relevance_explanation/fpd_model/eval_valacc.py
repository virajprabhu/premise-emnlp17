import sys
import json
import torchfile
import numpy as np

def compute_acc(tfile):
	tfile = torchfile.load(tfile)
	scores = tfile['scores']
	labels = tfile['labels']
	preds = np.argmax(scores, axis=1)
	preds += 1
	acc = float((preds == labels).sum())/float(len(labels))
	print((preds == 1).sum())
	print((preds == 2).sum())
	return acc

def main():
	args = sys.argv[1:]
	print('val acc: '+str(compute_acc(args[0])))

if __name__ == '__main__':
	main()