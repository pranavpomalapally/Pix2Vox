encoder = Encoder(cfg)
decoder = Decoder(cfg)
refiner = Refiner(cfg)
merger = Merger(cfg)

cfg.CONST.WEIGHTS = pretrained/Pix2Vox-A-ShapeNet.pth
chechpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))

fix_checkpoint = {}
fix_checkpoint['encoder_state_dict'] = orderedDict((k.split('module.')[1:][0], v) for k, v in checkpoint['encoder_state_dict'].items())
fix_checkpoint['decoder_state_dict'] = orderedDict((k.split('module.')[1:][0], v) for k, v in checkpoint['decoder_state_dict'].items())

epoch_idx = checkpoint['epoch_idx']
encoder.load_state_dict(fix_checkpoint['encoder_state_dict'])
decoder.load_state_dict(fix_checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()
refiner.eval()
merger.eval()

img1_path = '/datasets/ShapeNetRendering/02691156/1a04e3eab45ca15dd86060f189eb133/rendering/00.png'
img1_np = np.asarray(Image.open(img1_path))

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
    raw_features, generated_volum = decoder(image_features)

    if cfg.NETWORK.USE_MERFER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
        generated_volume = refiner(generated_volume)


generated_volume = generated_volume.squeeze(0)

img_dir= '/output/myresults'
gv = generated_volume.cpu().numpy()
rendering_views = utils.binvox_visualizatino.get_volume_views(gv, os.path.join(img_dir), epoch_idx)