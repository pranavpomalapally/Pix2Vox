import torch
ckpt = torch.load('/Users/pranavpomalapally/downloads/Pix2Vox-F-ShapeNet.pth', map_location=torch.device('cpu'))
new_ckpt = {}
new_ckpt['epoch_idx'] = ckpt['epoch_idx']
for k1 in ['encoder_state_dict', 'decoder_state_dict', 'merger_state_dict']:
# for k1 in ['encoder_state_dict', 'decoder_state_dict','refiner_state_dict', 'merger_state_dict']:
	new_ckpt[k1] = {}
	for k2 in ckpt[k1].keys():
  		new_ckpt[k1][k2.replace('module.', '')] = ckpt[k1][k2]

torch.save(new_ckpt, '/Users/pranavpomalapally/downloads/new-Pix2Vox-F-ShapeNet.pth')