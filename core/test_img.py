import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

from collections import OrderedDict
import cv2

def test_img(cfg):

	encoder = Encoder(cfg)
	decoder = Decoder(cfg)
	refiner = Refiner(cfg)
	merger = Merger(cfg)

	cfg.CONST.WEIGHTS = '/Users/pranavpomalapally/Downloads/new-Pix2Vox-A-ShapeNet.pth'
	checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))

	print()
	# fix_checkpoint = {}
	# fix_checkpoint['encoder_state_dict'] = OrderedDict((k.split('module.')[1:][0], v) for k, v in checkpoint['encoder_state_dict'].items())
	# fix_checkpoint['decoder_state_dict'] = OrderedDict((k.split('module.')[1:][0], v) for k, v in checkpoint['decoder_state_dict'].items())
	# fix_checkpoint['refiner_state_dict'] = OrderedDict((k.split('module.')[1:][0], v) for k, v in checkpoint['refiner_state_dict'].items())
	# fix_checkpoint['merger_state_dict'] = OrderedDict((k.split('module.')[1:][0], v) for k, v in checkpoint['merger_state_dict'].items())

	# fix_checkpoint['encoder_state_dict'] = OrderedDict((k.split('module.')[0], v) for k, v in checkpoint['encoder_state_dict'].items())
	# fix_checkpoint['decoder_state_dict'] = OrderedDict((k.split('module.')[0], v) for k, v in checkpoint['decoder_state_dict'].items())
	# fix_checkpoint['refiner_state_dict'] = OrderedDict((k.split('module.')[0], v) for k, v in checkpoint['refiner_state_dict'].items())
	# fix_checkpoint['merger_state_dict'] = OrderedDict((k.split('module.')[0], v) for k, v in checkpoint['merger_state_dict'].items())

	epoch_idx = checkpoint['epoch_idx']
	# encoder.load_state_dict(fix_checkpoint['encoder_state_dict'])
	# decoder.load_state_dict(fix_checkpoint['decoder_state_dict'])
	encoder.load_state_dict(checkpoint['encoder_state_dict'])
	decoder.load_state_dict(checkpoint['decoder_state_dict'])

	# if cfg.NETWORK.USE_REFINER:
	#  print('Use refiner')
	#  refiner.load_state_dict(fix_checkpoint['refiner_state_dict'])

	print('Use refiner')
	refiner.load_state_dict(checkpoint['refiner_state_dict'])
	if cfg.NETWORK.USE_MERGER:
		print('Use merger')
		# merger.load_state_dict(fix_checkpoint['merger_state_dict'])
		merger.load_state_dict(checkpoint['merger_state_dict'])


	encoder.eval()
	decoder.eval()
	refiner.eval()
	merger.eval()

	#img1_path = '/Users/pranavpomalapally/Downloads/ShapeNetRendering/02691156/1a04e3eab45ca15dd86060f189eb133/rendering/00.png'
	img1_path = '/Users/pranavpomalapally/Downloads/09 copy.png'
	img1_np = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

	sample = np.array([img1_np])

	IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
	CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

	test_transforms = utils.data_transforms.Compose([
	 utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
	 utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
	 utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
	 utils.data_transforms.ToTensor(),
	])

	rendering_images = test_transforms(rendering_images=sample)
	rendering_images = rendering_images.unsqueeze(0)

	with torch.no_grad():
	 image_features = encoder(rendering_images)
	 raw_features, generated_volume = decoder(image_features)

	 if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
	     generated_volume = merger(raw_features, generated_volume)
	 else:
	     generated_volume = torch.mean(generated_volume, dim=1)

	 # if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
	 #     generated_volume = refiner(generated_volume)
	generated_volume = refiner(generated_volume)
	generated_volume = generated_volume.squeeze(0)

	img_dir = '/Users/pranavpomalapally/Downloads/outputs'
	# gv = generated_volume.cpu().numpy()
	gv = generated_volume.cpu().detach().numpy()
	gv_new = np.swapaxes(gv, 2, 1)
	

	os.environ['KMP_DUPLICATE_LIB_OK']='True'
	rendering_views = utils.binvox_visualization.get_volume_views(gv_new, img_dir, epoch_idx)
