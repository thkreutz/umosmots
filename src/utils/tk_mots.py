
import numpy as np
import getpy as gp
import torch

class MOTS_Utils():
    def __init__(self, radius=1):
        self.d = self.init_radius_mat(radius)

    def init_radius_mat(self, radius, uniform=True):
        # neighbord radius, neighborhood is simple +- of coordinates from center point
        if uniform:
            r = radius
            dx = [-r, r]
            dy = [-r, r]
            dz = [-r, r]
        
        # could implement a non-uniform radius.
        else:
            pass
            #dx = self.rx
            #dy = self.ry
            #dz = self.rz
            
        d1 = np.linspace(dx[0], dx[1], num=np.sum(np.abs(dx)) + 1, endpoint=True)
        d2 = np.linspace(dy[0], dy[1], num=np.sum(np.abs(dy)) + 1, endpoint=True)
        d3 = np.linspace(dz[0], dz[1], num=np.sum(np.abs(dz)) + 1, endpoint=True)

        # cartesian
        return torch.Tensor(np.array(np.meshgrid(d1, d2, d3)).T.reshape(-1, 3).astype(int))
  
    def mts_dict_fast(self, frame):
        s = frame.long()
        offset = (-1 * torch.min(s[:,1:], dim=0)[0])
        s[:,1:] += offset  # minimum coord to be (0,0,0) for indexing purposes (cant index negative values in array and in hash map negative values also lead to problems.)
        shape = tuple(torch.max(s, dim=0)[0]+1) # we make h x w x d x t grid. find shape -> max coordinate in each dimension
        zeros = torch.zeros((shape[1], shape[2], shape[3], shape[0])) # make zeros
        zeros[s[:,1], s[:,2], s[:,3], s[:,0]] = 1 # index tensor by 4d coordinate and set to 1 to get time series of each voxel.
        return zeros, offset

    def to_mts_fast(self, frame, wsize=10):
        zeros, offset = self.mts_dict_fast(frame) # -> Time series tensor of each voxel
        x,y,z = torch.where(torch.sum(zeros, axis=3) > 0) # find occupied voxels 
        tss = zeros[x, y, z] # get the time series of ALL occupied voxels within this time frame. even if they where occupied only once
        vxls = torch.column_stack((x, y, z)).numpy() # get voxel coordinates 
        voxels = vxls - offset.numpy()  # recover original voxel coordinates by subtracting offset (see mts_dict_fast)
        #idxs = np.arange(len(vxls))

        mask = tss.bool()[:, -1] # select last occupancy value of each time series
        #mask = tss.bool()[:, int(wsize / 2) - 1] # use middle of window as center point to counter border and occlusion problem (maybe)
        vxls_masked = torch.Tensor(vxls[mask]) # get only occupied voxels at last timestep 
        voxels_masked = torch.Tensor(voxels[mask]) # original voxels 

        rep = vxls_masked.repeat_interleave(len(self.d), dim=0) # -> Like this we compute neighborhood, repeat each voxel d times
        nbors = rep + self.d.repeat(len(vxls_masked), 1) # -> add neighborhood kernel -> now we have the indices of each voxel + its neighorhood
        
        ### Parallel-Hashmap
        key_type = np.dtype('S16')
        value_type = np.dtype('S16')
        
        ### TODO: increase the sequence length up to 64 bit [X] done
        
        # bitpack coords into S16, each coord represented by int32, pad one zero to view S16
        vxls_temp = np.column_stack((np.zeros(len(vxls)), vxls)).astype(np.dtype('int32'))
        keys = vxls_temp.view('S16')
        
        # we start with bool type
        bits_temp = tss.bool().numpy().astype(np.dtype(bool))
        ## pad bool zeros up to 64 bit
        pad_diff = 64 - wsize
        values_temp = np.column_stack((bits_temp, np.zeros((len(bits_temp), pad_diff)).astype(bool)))
        # packbits to get 64 bit sequence
        values_temp = np.packbits(values_temp, axis=1).astype(np.uint16)
        # bitpack into s16
        #print("pad_diff=%s, bits_temp=%s -> values_temp=%s " % (pad_diff, bits_temp.shape, values_temp.shape) )
        values = values_temp.view('S16')
        
        # default return must be all zeros
        default_val = np.zeros(64).astype(bool) 
        default_val = np.packbits(default_val, axis=None).astype(np.uint16).view('S16') # no axis because its single value
        gp_dict = gp.Dict(key_type, value_type, default_value=default_val)
        gp_dict[keys] = values
        
        # neighborhood query
        nbors_temp = np.column_stack((np.zeros(len(nbors)), nbors)).astype(np.dtype('int32'))
        
        # parallel hashmap access
        result = gp_dict[nbors_temp.view('S16')]
        mts = torch.Tensor(np.unpackbits(result.view(np.uint16).astype(np.uint8), axis=1)[:, :wsize].astype('int'))
        
        # without parallel hashmap -> have to loop through a normal dict for each query -> SLOW
        #empty = np.zeros(wsize)
        #mts = torch.Tensor([ts_dict.get(tuple(key.tolist()), empty) for key in nbors])   

        # reshape into shape (n_voxels, n_channels, window_size)
        mts = mts.reshape(len(vxls_masked), len(self.d), wsize)
        
        # returns pair (features, voxels)
        return mts, voxels_masked



