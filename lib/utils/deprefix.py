import argparse
import torch
import pdb

parser = argparse.ArgumentParser(description="Remove PyTorch Model Prefix")
parser.add_argument('src_model', type=str)
parser.add_argument('dst_model', type=str)

args = parser.parse_args()
state_dict = torch.load(args.src_model, map_location=lambda storage, loc: storage)
state_dict = state_dict['state_dict']
pdb.set_trace()
state_dict = {('.'.join(k.split('.')[1:]) if "module" in k else k): v for k, v in state_dict.items()}
state_dict = {('.'.join(k.split('.')[1:]) if "base_model" in k else k): v for k, v in state_dict.items()}
torch.save(state_dict, args.dst_model)
