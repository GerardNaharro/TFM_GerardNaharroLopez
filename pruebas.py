import scipy.stats

p = scipy.stats.norm((100.799,50.451), 20).pdf((125,75))
pt = p[0] + p[1]
print("p 0: ", p[0] , ", p 1: ", p[1])
xd, xd2 = (10,36)
print(pt)
print(xd)
print(xd2)