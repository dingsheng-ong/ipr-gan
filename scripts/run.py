from PIL import Image
import caffe
import json
import labels
import numpy as np
import os
import scipy.misc
import sys

ID2TRAINID = {label.id: label.trainId for label in labels.labels}

def main():
	caffe.set_device(0)
	caffe.set_mode_gpu()

	net = caffe.Net(
		'caffemodel/deploy.prototxt',
		'caffemodel/fcn-8s-cityscapes.caffemodel',
		caffe.TEST
	)

	# classes = [
	# 	'road', 'sidewalk', 'building', 'wall', 'fence',
	# 	'pole', 'traffic light', 'traffic sign', 'vegetation',
	# 	'terrain', 'sky', 'person', 'rider', 'car', 'truck',
	# 	'bus', 'train', 'motorcycle', 'bicycle'
	# ]
	# num_classes = len(classes)
	num_classes = 19
	hist_perframe = np.zeros((num_classes, num_classes))

	input_dir = 'log/samples/cityscapes'
	label_dir = 'gt/'

	metrics_path = 'log/metrics.json'
	metrics = json.load(open(metrics_path))
	for i in range(500):
		# show progress
		sys.stdout.write('Evaluating: [%d/500]\r' % (i + 1))
		sys.stdout.flush()

		label_im = np.array(Image.open(os.path.join(label_dir, '%d.png' % i)))
		input_im = np.array(Image.open(os.path.join(input_dir, '%d.png' % i)))

		label_im = assign_label(label_im)
		label_im = label_im[np.newaxis, ...]

		input_im = scipy.misc.imresize(input_im, label_im.shape[1:])
		input_im = preprocess(input_im)
		output = segrun(net, input_im)

		hist_perframe += fast_hist(
			label_im.flatten(),
			output.flatten(),
			num_classes
		)
	
	pixel_acc, class_acc, class_iou, _, _ = get_scores(hist_perframe)
	sys.stdout.write('Mean pixel accuracy: %.4f\n' % pixel_acc)
	sys.stdout.write('Mean class accuracy: %.4f\n' % class_acc)
	sys.stdout.write('Mean class IoU: %.4f\n' % class_iou)
	sys.stdout.flush()
	
	metrics['cityscapes']['PIXEL_ACC'] = '%.4f' % pixel_acc
	metrics['cityscapes']['CLASS_ACC'] = '%.4f' % class_acc
	metrics['cityscapes']['CLASS_IOU'] = '%.4f' % class_iou
	json.dump(metrics, open(metrics_path, 'w'), indent=2, sort_keys=True)

def assign_label(label):
	label = np.array(label, dtype=np.float32)
	if sys.version_info[0] < 3:
		for k, v in ID2TRAINID.iteritems():
			label[label == k] = v
	else:
		for k, v in ID2TRAINID.items():
			label[label == k] = v
	return np.array(label, dtype=np.uint8)

def preprocess(image):
	image = np.array(image, dtype=np.float32)
	image = image[:, :, ::-1] # RGB -> BGR
	image -= np.array((72.78044, 83.21195, 73.45286))
	image = image.transpose((2, 0, 1)) # HWC -> CHW
	return image

def segrun(net, image):
	net.blobs['data'].reshape(1, *image.shape)
	net.blobs['data'].data[...] = image
	net.forward()
	return net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)

def fast_hist(a, b, n):
	k = np.where((a >= 0) & (a < n))[0]
	bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)
	if len(bc) != n ** 2: return 0
	return bc.reshape(n, n)

def get_scores(hist):
	acc = np.diag(hist).sum() / (hist.sum() + 1e-12)
	cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)
	iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)
	return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu

if __name__ == '__main__':
	main()