import numpy as np
from sklearn.model_selection import train_test_split

a = np.arange(1,101)
b = np.arange(501,601)
print(a.shape,b.shape)

a_split,a_test,b_split,b_test = train_test_split(a,b,test_size = 0.2,random_state=42)

print(a_split,'\n',a_test,'\n',b_split,'\n',b_test)