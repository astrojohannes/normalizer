import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import label

def exp_mask(mask_in, 
             iters = None, 
             constraint=None, 
             radius = None, 
             twoD = False, 
             no_edges = None, 
             keep_mask = False, 
             all_neighbours = False,
             quiet = False):
    """
    :param mask_in: required input: nd-array mask to be expanded
    :param iters: number of iterations to dilate the mask
    :param constraint: another mask to grow into (usually a lower S/N mask
    :param radius: radius in pixel to increase/grow the mask
    :param twoD: boolean, if True, expand mask only in image directions
    :param no_edges: 
    :param keep_mask: ensure that the original mask is part of the final mask
    :param all_neighbours:define diagonals to be adjacent pixels
    :param quiet: If True, suppress printing of informative lines
    """


#------------------------------------------------------------
# Default and Error Catching
#------------------------------------------------------------

    if not radius is None and (not iters is None or not constraint is None):
        raise ValueError("radius cannot be used with iters and/or contraint")
    
    if np.nansum(mask_in.flatten())==0:
        raise TypeError("Mask is empty, please provide a valid mask.")

#------------------------------------------------------------
# Mask initialization
#------------------------------------------------------------
	
    #initialize the output
    mask_out = copy.deepcopy(mask_in)
	
    # measure the dimensions of the mask
    mask_dim = np.shape(mask_out)

    # if we have an additional enclosure of mask
    if len(mask_dim)>3:
        mask_dim = np.shape(mask_out[0])
#------------------------------------------------------------
# Only a constraint is applied
#------------------------------------------------------------

#ToDo: Include no_edges and keep_mask options


    if not constraint is None and iters is None:
        """
            0 = no constraint
            1 = constraint
            2 = mask and constraint
        """
        if quiet == False:
            print("[INFO]\t Constraint mask provided.")
        mask = (mask_in*constraint)+constraint
	
        regions = label(mask)[0] 
		
        # Quit if we are empty
        if sum(regions.flatten())==0:
            raise TypeError("Mask is empty, please provide a valid mask.")
	
        #include any region that includes at leat one 2 in the mask 
        regs, regs_cts = np.unique(regions, return_counts=True)
        for i in range(len(regs-1)):
            inds = np.where(regions == i+1)
            if sum(mask[inds]==2)>0:
                mask_out[inds] = 1

        if keep_mask:
            mask_in = np.array(mask_in, dtype = int)
            mask_out = np.array(mask_out, dtype = int)

            mask_out = mask_in | mask_out
        if quiet == False:
            print("[DONE]  \t Growing mask using constraint mask finished, no errors. ")
        return mask_out


#------------------------------------------------------------
# Grow via iterations, with or without constraint
#------------------------------------------------------------

    if not iters is None:
        
        mask = copy.deepcopy(mask_in)
        
        # mask after an iteration to be used for a further iteration
        mask_iteration = copy.deepcopy(mask_in)
        mask_iteration = np.array(mask_in, dtype=int)
        
        if constraint is None:
            constraint = np.ones_like(mask)
            if quiet == False:
                print("[INFO]\t Iterations parameter provided.")
        else:
            if quiet == False:
                print("[INFO]\t Iteration parameter and constraint mask provided.")

		
        #make sure that mask, & constraint are binary type arrays
        mask = np.array(mask, dtype=int)
        constraint = np.array(constraint, dtype = int)
        # check the dimesnions of the cube:

        #1D:
        if len(mask_dim) == 1:
            if quiet == False:
                print("[INFO]\t Growing 1D mask via iterations.")
            for i in range(iters):
                mask = mask | np.roll(mask, -1)*constraint | np.roll(mask_in,+1)*constraint
                mask_iteration = copy.deepcopy(mask)

        #2D:
        elif len(mask_dim) == 2:
            if quiet == False:
                print("[INFO]\t Growing 2D mask via iterations.")
            for i in range(iters):
                for ii in range(-1,2):
                    for jj in range(-1,2):
                        if ii==0 or jj==0:
                            mask = mask | np.roll(mask_iteration, [ii,jj], axis = (0,1))*constraint
                        else:
                            #if all neighbours activated
                            if all_neighbours:
                                mask = mask | np.roll(mask_iteration, [ii,jj], axis = (0,1))*constraint

                mask_iteration = copy.deepcopy(mask)

        #3D:
        elif len(mask_dim) == 3:
            if quiet == False:
                print("[INFO]\t Growing 3D mask, via iterations.")
            for i in range(iters):
                for ii in range(-1,2):
                    for jj in range(-1,2):
						
                        # do not include the channels
                        if twoD:
                            if ii==0 or jj==0:
                                mask = mask | np.roll(mask_iteration, [ii,jj], axis = (1,2))*constraint
                            else:
                                #if all neighbours activated
                                if all_neighbours:
                                    mask = mask | np.roll(mask_iteration, [ii,jj], axis = (0,1))*constraint

                        else:
                            for zz in range(-1,2):
                                if sum([ii==0,jj==0,zz==0])==2:
                                    mask = mask | (np.roll(mask_iteration, [ii,jj,zz], axis = (0,1,2))*constraint)
                                else:
                                    if all_neighbours:
                                        mask = mask | (np.roll(mask_iteration, [ii,jj,zz], axis = (0,1,2))*constraint)
                mask_iteration = copy.deepcopy(mask)
		
        if keep_mask:
            mask_in = np.array(mask_in, dtype = int)
            mask = mask | mask_in
        if quiet == False:
            print("[DONE]\t Growing mask via iterations finished, no errors. ")
        return mask

#------------------------------------------------------------
# Grow radially
#------------------------------------------------------------
    if not radius is None:
        if quiet == False:
            print("[INFO]\t Radius parameter provided.")

        #take the smallest integer that is bigger than the input radius
        radius = np.floor(radius)
        radius = int(radius)
        
        #make sure that the mask is bonary type
        mask_in = np.array(mask_in, dtype = int)
        mask_out = np.array(mask_out, dtype = int)
        #vectorize the mask edges, defined as pixels that are 1 but border 0

        if len(mask_dim) == 1:
            if quiet == False:
                print("[INFO]\t Radially growing 1D mask.")
            edges = np.where(mask_in & ((np.roll(mask_in,1)+np.roll(mask_in,-1))!=2))[0]
            edge_cnt = len(edges)

        elif len(mask_dim) == 2:
            if quiet == False:
                print("[INFO]\t Radially growing 2D mask.")
            edges = np.where(mask_in & ((np.roll(mask_in,1, axis = 0)+
                                         np.roll(mask_in,-1, axis = 0)+
                                         np.roll(mask_in,1, axis = 1)+
                                         np.roll(mask_in,-1, axis = 1)
                                         )!=4))

            edge_cnt = len(edges)

        elif len(mask_dim) == 3:
            if quiet == False:
                print("[INFO]\t Radially growing 3D mask.")
            edges = np.where(mask_in & ((np.roll(mask_in,1, axis = 0)+
                                         np.roll(mask_in,-1, axis = 0)+
                                         np.roll(mask_in,1, axis = 1)+
                                         np.roll(mask_in,-1, axis = 1)+
                                         np.roll(mask_in,1, axis = 2)+
                                         np.roll(mask_in,-1, axis = 2)
                                         )!=6))

            edge_cnt = len(edges)
        else:
            raise ValueError("Not proper cube dimensions")

        if edge_cnt==0:
            if quiet == False:
                print("[INFO] \t No edges in the cube.")
                print("[INFO] \t Returning input mask.")
            return mask_in

        # 1D case
		
        if len(mask_dim) == 1:
            x = edges
            for i in range(-radius,radius+1):
                if i*i <= radius**2:
                    newx = copy.deepcopy(x)+i
                    newx[np.where(x+i<0)] = 0
                    newx[np.where(x+i>mask_dim[0]-1)] = mask_dim[0]-1
                
                    mask_out[newx] = 1

        # 2D case
        if len(mask_dim) == 2:
            x = edges[0]
            y = edges[1]
            for i in range(-radius,radius+1):
                for j in range(-radius,radius+1):
                    if (i**2 +j**2)<=radius**2:
                        newx = copy.deepcopy(x)+i
                        newx[np.where(x+i<0)] = 0
                        newx[np.where(x+i>mask_dim[0]-1)] = mask_dim[0]-1
                    
                        newy = copy.deepcopy(y)+j
                        newy[np.where(y+j<0)] = 0
                        newy[np.where(y+j>mask_dim[1]-1)] = mask_dim[1]-1
                    
                        mask_out[(newx,newy)] = 1

        # 3D case
        if len(mask_dim) == 3:
            x = edges[0]
            y = edges[1]
            z = edges[2]
            for i in range(-radius,radius+1):
                for j in range(-radius,radius+1):
                    if twoD:
                        if (i**2 +j**2)<=radius**2:
                            newx = copy.deepcopy(x)+i
                            newx[np.where(x+i<0)] = 0
                            newx[np.where(x+i>mask_dim[0]-1)] = mask_dim[0]-1

                            newy = copy.deepcopy(y)+j
                            newy[np.where(y+j<0)] = 0
                            newy[np.where(y+j>mask_dim[1]-1)] = mask_dim[1]-1

                        mask_out[(newx,newy,z)] = 1
                    else:
                        for k in range(-radius,radius+1):
                            if (i**2 +j**2+k**2)<=radius**2:
                                newx = copy.deepcopy(x)+i
                                newx[np.where(x+i<0)] = 0
                                newx[np.where(x+i>mask_dim[0]-1)] = mask_dim[0]-1
                        
                                newy = copy.deepcopy(y)+j
                                newy[np.where(y+j<0)] = 0
                                newy[np.where(y+j>mask_dim[1]-1)] = mask_dim[1]-1
                        
                                newz = copy.deepcopy(z)+k
                                newz[np.where(z+k<0)] = 0
                                newz[np.where(z+k>mask_dim[2]-1)] = mask_dim[2]-1
                        
                                mask_out[(newx,newy, newz)] = 1

        # always keep the mask for this case
        mask_out = mask_out | mask_in		
        if quiet == False:
            print("[DONE] \t Radially growing mask finished, no errors. ")
        return mask_out

#------------------------------------------------------------
# else, something was missing or incorrect
#------------------------------------------------------------
    
    
    raise TypeError("Incorrect Input for exp_mask function!")		

