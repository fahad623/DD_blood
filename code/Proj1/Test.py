import numpy as np
print np.arange(0.05, 0.25, 0.01)

    
arr = np.array([0.3, 0.5, 0.7])

arr[arr >= 0.5] = 1
arr[arr < 0.5] = 0