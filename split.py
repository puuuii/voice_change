f = open("1.txt", 'r', encoding='utf-8')
lines = f.readlines()
f.close()

for i, line in enumerate(lines):
    name = ('%04d' % (i+1)) + ".txt"
    f = open(name, 'w', encoding='utf-8')
    f.write(line)
    f.close()

