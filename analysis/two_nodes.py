import src.shuffle as st
import matplotlib.pyplot as plt
import numpy as np

diff_high = []; diff_low = []


for n in range(100):
    high1 = np.random.uniform(0,100,size=10)
    high2 = np.random.uniform(0,100,size=10)
    diff_high.append(st.diff_vectors(high1,high2))

    low1 = np.random.uniform(0,100,size=1)
    low2 = np.random.uniform(0,100,size=1)
    diff_low.append(st.diff_vectors(low1,low2))

p1 = plt.hist([np.average(n) for n in diff_high],alpha=0.5,bins=50,label='high')
p2 = plt.hist([np.average(n) for n in diff_low],alpha=0.5,bins=50, label='low')
plt.legend()
plt.show()