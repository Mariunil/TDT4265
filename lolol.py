import numpy as np
import pickle
import json

a = np.array([1,2,3,4,1,2,3,2,1,122.22])

file1 = open("model3_test_acc.txt", "w")
json.dump([1,2,3,4,1,2,3,2,1,122.22], file1)
file1.close()

file1 = open("model3_test_acc.txt", "r")
data = json.load(file1)
print(data)
file1.close()
