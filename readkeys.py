keys = list()
f = open('amitkeys.txt', 'r')
for line in f:
	raw = line.lower().replace("\n","").split(" ")
	keys.append(raw)
print (keys)