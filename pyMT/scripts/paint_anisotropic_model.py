import pyMT.data_structures as DS
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter

# Rewrite this to paint in the anisotropic WS model
# Test 1 - Extend the anisotropic region to include the tail.
# Test 2 - Should it extend closer to Nipigon?
# Test 3 - Can we simplifiy the rest of the model? e.g., turn the craton core 1D?

#########



orig_model = DS.Model('E:/phd/NextCloud/data/synthetics/EM3DANI/wst/wstZK_s1_nest.mod', file_format='em3dani')
model = deepcopy(orig_model)
model.rho_y = deepcopy(model.rho_x)
model.rho_z = deepcopy(model.rho_x)
rho_x = 10
rho_y = 3000
rho_z = 10

# First put in the upper mantle layer
# model.rho_x[66:96, 29:72,50:55] = 1000
# model.rho_y[66:96, 29:72,50:55] = 1000
# model.rho_z[66:96, 29:72,50:55] = 1000
# Put in the central block anisotropic block
# model.rho_x[66:96, 29:72,55:62] = rho_x
# model.rho_y[66:96, 29:72,55:62] = rho_y
# model.rho_z[66:96, 29:72,55:62] = rho_z
# Put in an anisotropic block under the Superior tail
for ix in range(36,96):
	for iy in range(20,80):
		if (iy > 38):
			if (ix < (iy)):
				continue
		# Upper mantle layer
		model.rho_x[ix, iy, 50:55] = 3000
		model.rho_y[ix, iy, 50:55] = 3000
		model.rho_z[ix, iy, 50:55] = 3000
		# Mid-mantle anisotropy
		# model.rho_x[ix, iy, 55:62] = rho_x
		# model.rho_y[ix, iy, 55:62] = rho_y
		# model.rho_z[ix, iy, 55:62] = rho_z
		####################################
		# For abrupt changes
		# Aniso7u
		# model.rho_x[ix,iy,55:60] = 100
		# model.rho_y[ix,iy,55:60] = 3000
		# model.rho_z[ix,iy,55:60] = 100
		# model.rho_x[ix,iy,60:64] = 5
		# model.rho_y[ix,iy,60:64] = 1000
		# model.rho_z[ix,iy,60:64] = 5
		# model.rho_x[ix,iy,64:66] = 50
		# model.rho_y[ix,iy,64:66] = 50
		# model.rho_z[ix,iy,64:66] = 50
		# Aniso7y
		# #100-146km
		# model.rho_x[ix,iy,55:58] = 300  
		# model.rho_y[ix,iy,55:58] = 3000
		# model.rho_z[ix,iy,55:58] = 300
		# #146-189
		# model.rho_x[ix,iy,58:60] = 100
		# model.rho_y[ix,iy,58:60] = 3000
		# model.rho_z[ix,iy,58:60] = 100
		# #189-244
		# model.rho_x[ix,iy,60:62] = 30
		# model.rho_y[ix,iy,60:62] = 1000
		# model.rho_z[ix,iy,60:62] = 30
		# #244-346
		# model.rho_x[ix,iy,62:64] = 10
		# model.rho_y[ix,iy,62:64] = 300
		# model.rho_z[ix,iy,62:64] = 10
		# #316-410
		# model.rho_x[ix,iy,64:66] = 3
		# model.rho_y[ix,iy,64:66] = 50
		# model.rho_z[ix,iy,64:66] = 3
		####################################
		# For gradual changes
		cc = 1
		for iz in range(55,66):
			# if ix > 55:
			# Set different log rho values for tail vs central blocks
			# if ix > 65:
			# 	starting_rho_x = np.log10(1000)
			# 	starting_rho_y = np.log10(3000)
			# 	step_size_x = 0.2
			# 	step_size_y = 0.1
			# else:
			# 	starting_rho_x = np.log10(1000)
			# 	starting_rho_y = np.log10(3000)
			# 	step_size_x = 0.2
			# 	step_size_y = 0.1
			# For depths < 215
			# Olivine-style, aniso8d, 9a
			if iz < 58:
				starting_rho_x = np.log10(3000)
				starting_rho_y = np.log10(3000)
				step_size_x = 0.25
				step_size_y = 0.0
				use_rho_y = 3000
			else:
				starting_rho_x = np.log10(3000)
				starting_rho_y = np.log10(3000)
				step_size_x = 0.25
				step_size_y = 0.1
				use_rho_y = 10**(starting_rho_y-(iz-58)*(step_size_y+0.01*cc))
			model.rho_x[ix, iy, iz] = max(10**(starting_rho_x-(iz-55)*(step_size_x+0.01*cc)), 5)
			model.rho_y[ix, iy, iz] = use_rho_y
			model.rho_z[ix, iy, iz] = max(10**(starting_rho_x-(iz-55)*(step_size_x+0.01*cc)), 5)
			cc += 1
			# # Graphite-style with decreasing rho_x, then reset to match rho_y
			# if iz < 58:
			# 	model.rho_x[ix, iy, iz] = 10**(2-(iz-55)*(0.2+0.01*cc))
			# 	model.rho_y[ix, iy, iz] = 3000
			# 	model.rho_z[ix, iy, iz] = 10**(2-(iz-55)*(0.2+0.01*cc))
			# elif iz < 60:
			# 	model.rho_x[ix, iy, iz] = 10**(2-(iz-55)*(0.2+0.01*cc)) #10
			# 	# model.rho_y[ix, iy, iz] = 3000
			# 	starting_rho_y = np.log10(3000)
			# 	step_size_y = 0.1
			# 	use_rho_y = 10**(starting_rho_y-(iz-58)*(step_size_y+0.005*cc))
			# 	model.rho_y[ix, iy, iz] = use_rho_y
			# 	model.rho_z[ix, iy, iz] = 10**(2-(iz-55)*(0.2+0.01*cc)) #10
			# else:
			# 	starting_rho_x = np.log10(use_rho_y)
			# 	starting_rho_y = np.log10(use_rho_y)
			# 	step_size_x = 0.1
			# 	step_size_y = 0.1
			# 	model.rho_x[ix, iy, iz] = 10**(starting_rho_x-(iz-59)*(step_size_x+0.01*cc))
			# 	model.rho_y[ix, iy, iz] = 10**(starting_rho_y-(iz-59)*(step_size_x+0.01*cc))
			# 	model.rho_z[ix, iy, iz] = 10**(starting_rho_x-(iz-59)*(step_size_x+0.01*cc))
			# # Aniso8b-c (Graphite to 189 then reset to normal mantle, decrease rho with depth)
			# if iz < 58:
			# 	model.rho_x[ix, iy, iz] = 100
			# 	model.rho_y[ix, iy, iz] = 3000
			# 	model.rho_z[ix, iy, iz] = 100
			# elif iz < 60:
			# 	model.rho_x[ix, iy, iz] = 10
			# 	model.rho_y[ix, iy, iz] = 3000
			# 	model.rho_z[ix, iy, iz] = 10
			# else:
			# 	model.rho_x[ix, iy, iz] = 10**(starting_rho_x-(iz-60)*(step_size_x+0.01*cc))
			# 	model.rho_y[ix, iy, iz] = 10**(starting_rho_y-(iz-60)*(step_size_x+0.01*cc))
			# 	model.rho_z[ix, iy, iz] = 10**(starting_rho_x-(iz-60)*(step_size_x+0.01*cc))
		# 	# model.rho_x[ix, iy, iz] = rho_x
		# 	# model.rho_y[ix, iy, iz] = rho_y
		# 	# model.rho_z[ix, iy, iz] = rho_z
		# 	# Less conductive in the tail?
		# 	# else:
		# 	# 	model.rho_x[ix, iy, iz] = max(10**(2.5-(iz-55)*0.25), 5)
		# 	# 	model.rho_y[ix, iy, iz] = rho_y
		# 	# 	model.rho_z[ix, iy, iz] = max(10**(2.5-(iz-55)*0.25), 5)
		# for iz in range(62,66):
		# 	model.rho_x[ix,iy,iz] = 5#15 - (iz-62)*2 #min(model.rho_x[ix,iy,iz], 300)
		# 	model.rho_y[ix,iy,iz] = 5#15 - (iz-62)*2 #min(model.rho_x[ix,iy,iz], 300)
		# 	model.rho_z[ix,iy,iz] = 5#15 - (iz-62)*2 #min(model.rho_x[ix,iy,iz], 300)
# Add more tail
for ix in range(28,80):
	for iy in range(10,38):
		# if ix > 38:
		# 	if iy < ix-38:
		# 		continue
		if iy < 20:
			if model.rho_x[ix, iy, 55] < 1000:
				continue
		# Add upper mantle piece in these parts too
		model.rho_x[ix, iy, 50:55] = 3000
		model.rho_y[ix, iy, 50:55] = 3000
		model.rho_z[ix, iy, 50:55] = 3000
		####################################
		# Aniso7u
		# model.rho_x[ix,iy,55:60] = 100
		# model.rho_y[ix,iy,55:60] = 3000
		# model.rho_z[ix,iy,55:60] = 100
		# model.rho_x[ix,iy,60:64] = 5
		# model.rho_y[ix,iy,60:64] = 1000
		# model.rho_z[ix,iy,60:64] = 5
		# model.rho_x[ix,iy,64:66] = 50
		# model.rho_y[ix,iy,64:66] = 50
		# model.rho_z[ix,iy,64:66] = 50
		# # Aniso7y
		# #100-146km
		# model.rho_x[ix,iy,55:58] = 300  
		# model.rho_y[ix,iy,55:58] = 3000
		# model.rho_z[ix,iy,55:58] = 300
		# #146-189
		# model.rho_x[ix,iy,58:60] = 100
		# model.rho_y[ix,iy,58:60] = 3000
		# model.rho_z[ix,iy,58:60] = 100
		# #189-244
		# model.rho_x[ix,iy,60:62] = 30
		# model.rho_y[ix,iy,60:62] = 1000
		# model.rho_z[ix,iy,60:62] = 30
		# #244-346
		# model.rho_x[ix,iy,62:64] = 10
		# model.rho_y[ix,iy,62:64] = 300
		# model.rho_z[ix,iy,62:64] = 10
		# #316-410
		# model.rho_x[ix,iy,64:66] = 3
		# model.rho_y[ix,iy,64:66] = 50
		# model.rho_z[ix,iy,64:66] = 3
		# Lower isotropic block
		# model.rho_x[ix,iy,64:66] = 50
		# model.rho_y[ix,iy,64:66] = 50
		# model.rho_z[ix,iy,64:66] = 50
		
		####################################
		# For gradual changes
		cc = 1
		for iz in range(55,66):
			# if ix > 55:
			# Set different log rho values for tail vs central blocks
			# # Aniso8b-c (Graphite to 189 then reset to normal mantle, decrease rho with depth)
			# if ix > 65:
			# 	starting_rho_x = np.log10(1000)
			# 	starting_rho_y = np.log10(3000)
			# 	step_size_x = 0.2
			# 	step_size_y = 0.1
			# else:
			# 	starting_rho_x = np.log10(1000)
			# 	starting_rho_y = np.log10(3000)
			# 	step_size_x = 0.2
			# 	step_size_y = 0.1
			# if iz < 58:
			# 	model.rho_x[ix, iy, iz] = 10**(2-(iz-55)*(0.2+0.01*cc))
			# 	model.rho_y[ix, iy, iz] = 3000
			# 	model.rho_z[ix, iy, iz] = 10**(2-(iz-55)*(0.2+0.01*cc))
			# elif iz < 60:
			# 	model.rho_x[ix, iy, iz] = 10**(2-(iz-55)*(0.2+0.01*cc)) #10
			# 	# model.rho_y[ix, iy, iz] = 3000
			# 	starting_rho_y = np.log10(3000)
			# 	step_size_x = 0.1
			# 	step_size_y = 0.1
			# 	use_rho_y = 10**(starting_rho_y-(iz-58)*(step_size_y+0.005*cc))
			# 	model.rho_y[ix, iy, iz] = use_rho_y
			# 	model.rho_z[ix, iy, iz] = 10**(2-(iz-55)*(0.2+0.01*cc)) #10
			# else:
			# 	starting_rho_x = np.log10(use_rho_y)
			# 	starting_rho_y = np.log10(use_rho_y)
			# 	model.rho_x[ix, iy, iz] = 10**(starting_rho_x-(iz-59)*(step_size_x+0.01*cc))
			# 	model.rho_y[ix, iy, iz] = 10**(starting_rho_y-(iz-59)*(step_size_x+0.01*cc))
			# 	model.rho_z[ix, iy, iz] = 10**(starting_rho_x-(iz-59)*(step_size_x+0.01*cc))
			# cc += 1
			#Aniso9a (change iz start to 59), as is its aniso9d
			if iz < 58:
				starting_rho_x = np.log10(3000)
				starting_rho_y = np.log10(3000)
				step_size_x = 0.25
				step_size_y = 0.0
				use_rho_y = 3000
			else:
				starting_rho_x = np.log10(3000)
				starting_rho_y = np.log10(3000)
				step_size_x = 0.25
				step_size_y = 0.1
				use_rho_y = 10**(starting_rho_y-(iz-58)*(step_size_y+0.01*cc))
			model.rho_x[ix, iy, iz] = max(10**(starting_rho_x-(iz-55)*(step_size_x+0.01*cc)), 5)
			model.rho_y[ix, iy, iz] = use_rho_y
			model.rho_z[ix, iy, iz] = max(10**(starting_rho_x-(iz-55)*(step_size_x+0.01*cc)), 5)
			cc += 1
		# 	# model.rho_x[ix, iy, iz] = rho_x
		# 	# model.rho_y[ix, iy, iz] = rho_y
		# 	# model.rho_z[ix, iy, iz] = rho_z
		# 	# Less conductive in the tail?
		# 	# else:
		# 	# 	model.rho_x[ix, iy, iz] = max(10**(2.5-(iz-55)*0.25), 5)
		# 	# 	model.rho_y[ix, iy, iz] = rho_y
		# 	# 	model.rho_z[ix, iy, iz] = max(10**(2.5-(iz-55)*0.25), 5)
		# for iz in range(62,66):
		# 	model.rho_x[ix,iy,iz] = 5# - (iz-62)*2 #min(model.rho_x[ix,iy,iz], 300)
		# 	model.rho_y[ix,iy,iz] = 5# - (iz-62)*2 #min(model.rho_x[ix,iy,iz], 300)
		# 	model.rho_z[ix,iy,iz] = 5# - (iz-62)*2 #min(model.rho_x[ix,iy,iz], 300)

# Put a mantle conductor in the west sitting against the archean lithosphere
# max_y = 38
# for iz in range(55,62):
for ix in range(28,96): # Was 96, but now try extending it up above the craton core
		# idx = np.argwhere(model.rho_y[ix,:, 55] == rho_y)[0][0]
		# max_y = idx + (iz-55)*3
	for iy in range(0,38): # Was 38, but have it bend around the craton at the top
		# 	model.rho_x[ix,iy,iz] = 10 #ix # have it decrease in resistivity going south
		# 	model.rho_y[ix,iy,iz] = 10 #ix # have it decrease in resistivity going south
		# 	model.rho_z[ix,iy,iz] = 10 #ix # have it decrease in resistivity going south
		if model.rho_y[ix,iy,54] != rho_y:
			model.rho_x[ix,iy,55:62] = 10 #ix # have it decrease in resistivity going south
			model.rho_y[ix,iy,55:62] = 10 #ix # have it decrease in resistivity going south
			model.rho_z[ix,iy,55:62] = 10 #ix # have it decrease in resistivity going south
# Continue the conductor to the south
model.rho_x[0:28,0:15,55:62] = 10
model.rho_y[0:28,0:15,55:62] = 10
model.rho_z[0:28,0:15,55:62] = 10

# # Extend the mantle conductor so it continues north and bends around the craton
# for ix in range(96,model.nx): # Was 96, but now try extending it up above the craton core
# 		# idx = np.argwhere(model.rho_y[ix,:, 55] == rho_y)[0][0]
# 		# max_y = idx + (iz-55)*3
# 	for iy in range(0,60): # Was 38, but have it bend around the craton at the top
# 		# 	model.rho_x[ix,iy,iz] = 10 #ix # have it decrease in resistivity going south
# 		# 	model.rho_y[ix,iy,iz] = 10 #ix # have it decrease in resistivity going south
# 		# 	model.rho_z[ix,iy,iz] = 10 #ix # have it decrease in resistivity going south
# 		if ix < 126 and iy > 20:
# 			continue
# 		if ix > 125 and iy > 20 + ix - 126:
# 			continue
# 		model.rho_x[ix,iy,55:62] = 10 #ix # have it decrease in resistivity going south
# 		model.rho_y[ix,iy,55:62] = 10 #ix # have it decrease in resistivity going south
# 		model.rho_z[ix,iy,55:62] = 10 #ix # have it decrease in resistivity going south

# # Make a 1D craton core
# for ix in range(96, model.nx-25):
# 	for iy in range(27, 65):
# 		for iz in range(49, 55):
# 			model.rho_x[ix,iy,iz] = np.max((model.rho_x[ix,iy,iz], 10000))
# 			model.rho_y[ix,iy,iz] = np.max((model.rho_y[ix,iy,iz], 10000))
# 			model.rho_z[ix,iy,iz] = np.max((model.rho_z[ix,iy,iz], 10000))
# step_size = 0.1
# for ix in range(96, model.nx-25):
# 	for iy in range(27,65):
# 		cc = 1
# 		if model.rho_x[ix,iy,55] != 10:
# 			for iz in range(55, 66):
# 				model.rho_x[ix,iy,iz] = 10**(np.log10(model.rho_x[ix,iy,iz-1]) - (step_size + 0.01*cc))
# 				model.rho_y[ix,iy,iz] = 10**(np.log10(model.rho_x[ix,iy,iz-1]) - (step_size + 0.01*cc))
# 				model.rho_z[ix,iy,iz] = 10**(np.log10(model.rho_x[ix,iy,iz-1]) - (step_size + 0.01*cc))
# 				cc += 1
# 				# std = np.std(model.rho_x[96:model.nx, 15:80,iz].flatten())
# 				# # median = np.median(model.rho_x[96:model.nx,15:80,iz].flatten())
# 				# mean = 10**np.mean(np.log10(model.rho_x[96:model.nx,15:80,iz].flatten()))
# 				# if model.rho_x[ix,iy,iz] < mean:
# 				# # model.rho_x[ix, iy, iz] = np.median(model.rho_x[96:model.nx,15:80,iz].flatten())
# 				# 	model.rho_x[ix, iy, iz] = mean
# 				# 	model.rho_y[ix, iy, iz] = mean
# 				# 	model.rho_z[ix, iy, iz] = mean
# # 			# # Upper mantle 1
# # 			# model.rho_x[ix, iy, 50:53] = 50000
# # 			# model.rho_y[ix, iy, 50:53] = 50000
# # 			# model.rho_z[ix, iy, 50:53] = 50000
# # 			# # Upper mantle 2
# # 			# model.rho_x[ix, iy, 53:56] = 5000
# # 			# model.rho_y[ix, iy, 53:56] = 5000
# # 			# model.rho_z[ix, iy, 53:56] = 5000
# # 			# # mid mantle
# # 			# model.rho_x[ix, iy, 56:59] = 1000
# # 			# model.rho_y[ix, iy, 56:59] = 1000
# # 			# model.rho_z[ix, iy, 56:59] = 1000

# # 			# model.rho_x[ix, iy, 59:61] = 300
# # 			# model.rho_y[ix, iy, 59:61] = 300
# # 			# model.rho_z[ix, iy, 59:61] = 300

# # 			# model.rho_x[ix, iy, 61:63] = 100
# # 			# model.rho_y[ix, iy, 61:63] = 100
# # 			# model.rho_z[ix, iy, 61:63] = 100

# # 			# model.rho_x[ix, iy, 63:64] = 50
# # 			# model.rho_y[ix, iy, 63:64] = 50
# # 			# model.rho_z[ix, iy, 63:64] = 50

# # 			# model.rho_x[ix, iy, 64] = 30
# # 			# model.rho_y[ix, iy, 64] = 30
# # 			# model.rho_z[ix, iy, 64] = 30

# Apply smoothing if the resistivities are extreme (maybe help with the crashing issue?)
# for iz in range(55,66):
# 	model.rho_x[:,:,iz] = gaussian_filter(model.rho_x[:,:,iz], [2,2])
# 	model.rho_y[:,:,iz] = gaussian_filter(model.rho_y[:,:,iz], [2,2])
# 	model.rho_z[:,:,iz] = gaussian_filter(model.rho_z[:,:,iz], [2,2])

model.strike = np.zeros(model.vals.shape)
model.strike[:74,:,54:66] = 20 # Make the tail piece have 20 degree strike
model.slant = np.zeros(model.vals.shape)
model.dip = np.zeros(model.vals.shape)
model.write('E:/phd/NextCloud/data/synthetics/EM3DANI/wst/aniso9/wstZK_aniso9d',
			file_format='em3dani',
			use_log=True,
			use_resistivity=True,
			use_anisotropy=True)
