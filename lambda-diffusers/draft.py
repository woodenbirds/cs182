import csv
csv_reader=csv.reader(open("full_emoji.csv"))
j=0
for row in csv_reader:
    for i in range(len(row)):
        print(i,row[i])
    j+=1
    if(j==2):
        break