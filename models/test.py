import os

directory = os.walk('E:\\')
lists = open('D:\\Music\\list.txt', 'w')
for dirs in directory:
    for file in dirs[2]:
        if '.mp3' in file:
            lists.writelines(file+'\n')
            print(file)