import numpy as np 

import matplotlib.pyplot as plt


im1=plt.imread("bird.jpg", format=None)

print(im1.shape)
# convert to grayscale image
im1g=0.2989 * im1[:,:,0] + 0.5870 * im1[:,:,1] + 0.1140  * im1[:,:,2] 
plt.imshow(im1g,cmap='gray')
plt.show()
#singular value decomposition
u, s, vh = np.linalg.svd(im1g, full_matrices=False)

maxk=len(s)
print(maxk)
S=np.diag(s)
ratio=.03 # 0 to 1
k=int(ratio*maxk)
print("maxk= ",maxk,"   k= ",k)
print("singular value : ",s.astype(int))



newim1g=u[:,0:k]@S[0:k,0:k]@vh[0:k,:]
    
fig=plt.figure()

fig.add_subplot(2,2,1)
plt.imshow(im1g,cmap='gray')

fig.add_subplot(2, 2,2)
plt.imshow(newim1g,cmap='gray')

fig.add_subplot(2, 2,3)
plt.imshow(S[0:k,0:k],cmap='gray')

fig.add_subplot(2, 2,4)
plt.imshow(abs(newim1g-im1g),cmap='gray')

plt.show()
   



