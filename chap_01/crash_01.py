# Python Crash Course

"""
*Data
types
*Numbers
*Strings
*Printing
*Lists
*Dictionaries
*Booleans
*Tuples
*Sets
*Comparison
Operators
* if, elif, else Statements
* for Loops
* while Loops
*range()
*list
comprehension
*functions
* lambda expressions
*map and filter
*methods
____
"""
## Data types

### Numbers
print(1 + 1)
print(1 * 3)
print(1 / 2)
print(2 ** 4)
print(4 % 2)
print(5 % 2)
print((2 + 3) * (5 + 5))

### Variable Assignment
# Can not start with number or special characters
name_of_var = 2
x = 2
y = 3
z = x + y
print(z)

# Multiple Assignment
a, b, c = 1, 2, 3
print(a, b, c)

### Strings
'single quotes'
"double quotes"
" wrap lot's of other quotes"

# Strings
data = 'hello world'
print(data[0])
print(len(data))
print(data)

### Printing
x = 'hello'
print(x)
num = 12
name = 'Sam'
print('My number is: {one}, and my name is: {two}'.format(one=num, two=name))
print('My number is: {}, and my name is: {}'.format(num, name))

### Lists
a = [1, 2, 3]
b = ['hi', 1, [1, 2]]
my_list = ['a', 'b', 'c']
my_list.append('d')
print(my_list)
print(my_list[0])
print(my_list[1])
print(my_list[1:])
print(my_list[:1])
my_list[0] = 'NEW'
print(my_list)
nest = [1, 2, 3, [4, 5, ['target']]]
print(nest[3])
print(nest[3][2])
print(nest[3][2][0])

### Dictionaries
d = {'key1': 'item1', 'key2': 'item2'}
print(d)
print(d['key1'])
print(d.get('key2'))
### Booleans
### True
### False

### Tuples
t = (1, 2, 3)
print(t[0])
# t[0] = 'NEW' -> 값 변경 X

### Sets
### 중복 제거
set1 = {1, 2, 3}
set2 = {1, 2, 3, 1, 2, 1, 2, 3, 3, 3, 3, 2, 2, 2, 1, 1, 2}
print(set1.union(set2))

## Comparison Operators
print(1 > 2)
print(1 < 2)
print(1 >= 1)
print(1 <= 4)
print(1 == 1)
print('hi' == 'bye')

## Logic Operators
print((1 > 2) and (2 < 3))
print((1 > 2) or (2 < 3))
print((1 == 2) or (2 == 3) or (4 == 4))

## if, elif,  else Statements
if 1 < 2:
    print('ok')
if 1 < 2:
    print('ok')
if 1 < 2:
    print('first')
else:
    print('last')
if 1 > 2:
    print('first')
else:
    print('last')
if 1 == 2:
    print('first')
elif 3 == 3:
    print('middle')
else:
    print('Last')

## for Loops
seq = ['a', 'b', 'c', 'd', 'e']
for item in seq:
    print(item)

## enumerate
## index, element
for i, el in enumerate(seq):
    print(i, el)

## while Loops
i = 1
while i < 5:
    print('i is: {}'.format(i))
    i = i + 1

## range()
range(5)
for i in range(5):
    print(i)
list(range(5))

## list comprehension
## compact for loop
x = [1, 2, 3, 4]
out = []
for item in x:
    out.append(item ** 2)
print(out)
c1 = [item ** 2 for item in x]
print(c1)

## functions
# Sum function
def mysum(x, y):
    sum = x + y
    return sum

# Test sum function
print(mysum(1, 3))

def square(x):
    return x ** 2
out = square(2)
print(out)

## lambda expressions
def times2(x):
    return x * 2
times2(2)
(lambda x: x * 2)(2)

## map and filter
seq = [1, 2, 3, 4, 5]
list(map(times2, seq))
list(map(lambda x: x * 2, seq))
filter(lambda x: x % 2 == 0, seq)
list(filter(lambda x: x % 2 == 0, seq))

## methods
st = 'hello my name is Sam'
st.lower()
st.upper()
st.split()
tweet = 'Go Sports! #Sports'
tweet.split('#')
print(tweet.split('#')[1])
print(d)
d.keys()
d.items()
lst = [1, 2, 3]
lst.pop()
print(lst)
print('x' in [1, 2, 3])
print('x' in ['x', 'y', 'z'])