import numpy as np

gape_array = np.load('/home/julia-corman/Desktop/Personal_Data/JC9/JC9_taste_day1_260313_125116/emg_output/gape_array.npy')
p_data = np.load('/home/julia-corman/Desktop/Personal_Data/JC11/JC11_taste_day_260225_131220/emg_output/p_data.npy')

print(np.shape(gape_array))
print(np.shape(p_data))

