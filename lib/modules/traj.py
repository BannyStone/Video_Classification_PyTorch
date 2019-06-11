import torch
import torch.nn as nn
import pdb

class CorrTrajBlock(nn.Module):
	"""
	New Project: 
	"""
	def __init__(self, input_dim, reduce_dim, topk=4):
		super(CorrTrajBlock, self).__init__()
		self.input_dim = input_dim
		self.reduce_dim = reduce_dim
		self.topk = topk
		self.conv_reduce_dim = nn.Conv1d(input_dim, 
										reduce_dim, 
										kernel_size=1, 
										bias=False)
		self.bn_reduce_dim = nn.BatchNorm1d(reduce_dim)
		# self.coordinates_proj = nn.Conv2d(2, input_dim//4,
		# 								kernel_size=1,
		# 								bias=False)
		# self.coordinates_proj_bn = nn.BatchNorm2d(input_dim//4)
		self.traj_proj = nn.Conv2d(input_dim+2, input_dim//4,
									kernel_size=1,
									bias=False)
		self.traj_proj_bn = nn.BatchNorm2d(input_dim//4)
		self.t_conv = nn.Conv2d(input_dim//4, input_dim,
								kernel_size=(3,1),
								padding=(1,0),
								bias=False)
		self.t_bn = nn.BatchNorm2d(input_dim)

		self.relu = nn.ReLU(inplace=True)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, input):
		assert(len(input.shape) == 5), "input shape must have 5 dimension" #[N,D,T,H,W]
		assert(input.shape[1] == self.input_dim)
		batch_size = input.shape[0]
		length = input.shape[2]
		height = input.shape[3]
		width = input.shape[4]
		# select template frame
		## Plan1: Select 1st frame
		template_d = input[:,:,0,:,:].view(-1, self.input_dim, height*width) #[N,D,HW]
		# reduce template channel dimension
		template_p = self.bn_reduce_dim(self.conv_reduce_dim(template_d)) #[N,P,HW]
		# Adaptive Sampling for Trajectory Starting Points
		_, spt_inds = template_p.max(dim=2)
		spt_inds = spt_inds.squeeze().view(batch_size, 1, self.reduce_dim).expand((batch_size, self.input_dim, self.reduce_dim)) #[N,D,P]

		# Feature Resampling
		template_resample = torch.gather(input=template_d, dim=2, index=spt_inds) #[N,D,P]
		affinity_matrix = torch.matmul(template_resample.transpose(1, 2), 
										input.view(batch_size, self.input_dim, length*height*width)) #[N,P,THW]
		_, topk_inds_per_frame = affinity_matrix.view(batch_size, 
													self.reduce_dim, 
													length, 
													height*width).topk(self.topk, dim=3) #[N,P,T,K]
		# pdb.set_trace()
		topk_inds_per_frame = topk_inds_per_frame.transpose(1,3) #[N,K,T,P]

		traj_features = []
		for k in range(self.topk):
			kth_per_frame = topk_inds_per_frame[:,k,:,:].view(
															batch_size, 
															1, 
															length, 
															self.reduce_dim).expand(
															batch_size, 
															self.input_dim, 
															length, 
															self.reduce_dim) #[N,D,T,P]
			kth_feature_sampling = torch.gather(input=input.view(batch_size, self.input_dim, length, height*width), 
												dim=3, 
												index=kth_per_frame) #[N,D,T,P]
			traj_features.append(kth_feature_sampling)

		traj_features = torch.cat(traj_features, dim=1).view(batch_size*self.topk, self.input_dim, length, self.reduce_dim) #[NK,D,T,P]
		with torch.no_grad():
			topk_inds_per_frame = topk_inds_per_frame.contiguous()
			# pdb.set_trace()
			row_inds = topk_inds_per_frame.view(batch_size*self.topk, 1, length, self.reduce_dim) // width
			col_inds = topk_inds_per_frame.view(batch_size*self.topk, 1, length, self.reduce_dim) % width
		coordinates = torch.cat([row_inds/height, col_inds/width], dim=1).float() #[NK,2,T,P]
		# coordinates_proj = self.relu(self.coordinates_proj_bn(self.coordinates_proj(coordinates)))
		# pdb.set_trace()
		fuse_traj_features = torch.cat([traj_features, coordinates], dim=1) #[NK,(D+2),T,P]
		fuse_traj_features = self.traj_proj_bn(self.traj_proj(fuse_traj_features)).view(batch_size, 
																						self.topk, 
																						self.input_dim//4, 
																						length, 
																						self.reduce_dim) #[N,K,D//4,T,P]
		fuse_traj_features, _ = torch.max(fuse_traj_features, dim=1)
		fuse_traj_features = fuse_traj_features.squeeze() #[N,D//4,T,P]
		traj_conv_features = self.relu(self.t_bn(self.t_conv(fuse_traj_features))) #[N,D,T,P]

		points_features = traj_features.view(batch_size, self.topk, self.input_dim, length*self.reduce_dim).mean(dim=1).squeeze() #[N,D,TP]
		proj_matrix = torch.matmul(points_features.transpose(1,2), input.view(batch_size, self.input_dim, length*height*width)) #[N,TP,THW]
		# TODO: Restrict proj matrix by measuring the distance between topk coordinates

		proj_matrix = self.softmax(proj_matrix)
		traj_conv_propagated_features = torch.matmul(
													traj_conv_features.view(batch_size, self.input_dim, length*self.reduce_dim), 
													proj_matrix).view(
													batch_size, self.input_dim, length, height, width) #[N,D,T,H,W]

		return input + traj_conv_propagated_features

if __name__ == "__main__":
	m = CorrTrajBlock(input_dim=256, reduce_dim=32, topk=4)
	input = torch.ones((64,256,8,56,56))
	output = m(input)
	pdb.set_trace()