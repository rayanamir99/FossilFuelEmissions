import numpy as np

with open("2019pollution.csv") as file:
    text = file.read()

# convert into list of lists
arr = text.split("\n")
mat = [i.split(",") for i in arr]

mat = mat[1:] #take out the first two headers

print('--------------------')
print("GOAL: find if there is an association between the \ncarbon monoxide output and the nitrogen oxide output by country")
print("if there is an association, what is the given ratio?")
print('--------------------')

# lists for nitrogen oxide (x) and carbon monoxide (y)
x = [i[1] for i in mat if len(i) > 2 and float(i[1])]
y = [i[2] for i in mat if len(i) > 2 and float(i[1])]

print("First 5 values of x: the nitrogen oxide \t" + str(x[0:5]))
print("First 5 values of y: the carbon monoxide \t" + str(y[0:5]))
print('--------------------')

# converts into floats and numpy arrays
t = np.array(x, dtype=float)
b = np.array(y, dtype=float)

# A matrix
A = np.vstack([t, np.ones(len(t))]).T
new_A = np.matmul(A.T, A)
new_B = np.matmul(A.T, b)

# solve for c and d values
x_hat = np.linalg.solve(new_A, new_B)

#still need: error value and projections

p = []
for i in t:
    p.append(i*x_hat[0]+x_hat[1])
print("first 5 vals of projection matrix: " + str(p[0:5]))
print('--------------------')

e = []
for i in range(len(p)):
    e.append(b[i] - p[i])

p = np.array(p,dtype = float)
e = np.array(e,dtype = float)

#print(np.matmul(p,e)) #final value of 1.818, with the rounding error not too bad

error = 0
for i in e:
    error += i**2
print("Total error: " + str(error) + "\nbig number for a big calculation, when dealing in the billions this is normal")
print('--------------------')

print("y intercept: " + str(x_hat[1]) + "\nslope: " + str(x_hat[0]))
