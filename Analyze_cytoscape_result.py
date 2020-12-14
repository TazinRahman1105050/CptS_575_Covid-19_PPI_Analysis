import csv
import math
row_num = 0
closeness=[]
degree=[]
eigenvector=[]
betweenness=[]
LACW=[]
information=[]
avg=[]
count = 0

#read file
with open('Result.csv.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
        #print(row[2])
        if count > 0:

            betweenness.append(float(row[2]))
            closeness.append(float(row[4]))
            degree.append(float(row[6]))
            eigenvector.append(float(row[9]))
            information.append(float(row[10]))
            LACW.append(float(row[12]))
            sum = float(row[2])+float(row[4])+float(row[6])+float(row[9])+float(row[10])+float(row[12])
            avg.append(sum/6)
        count = 1

#calculate average
tot =0.0
for i in range(0, len(avg)):
    print(avg[i])
    tot = tot + avg[i]
avg_tot = tot/len(avg)
tot = 0.0
for i in range(0, len(avg)):
    tot += (avg[i]-avg_tot)*(avg[i]-avg_tot)
tot = tot/len(avg)
std_dev = math.sqrt(tot)

#calculate z_score and classify hub and non-hub
z_score = []
hub = []
for i in range(0, len(avg)):
    score = (avg[i] - avg_tot)/std_dev
    z_score.append((avg[i] - avg_tot)/std_dev)
    if score >= 1:
       hub.append(1)
    else:
       hub.append(0)
counter= 0

#output the data in a csv file
with open('Hub1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Degree", "Closeness", "Betweenness", "Evcent", "Information", "LACW", "Avg", "hub"])


    for i in range(0, len(avg)):

        #avg = (degree_list[count]+close[count]+between[count]+evcentrality[count])/4.0
        writer.writerow([degree[counter], closeness[counter], betweenness[counter], eigenvector[counter], information[counter], LACW[counter], avg[counter], hub[counter]])
        counter += 1


        #print(str(row[2]) +" "+ str(row[4]) + " "+ str(row[6])+ " "+ str(row[9]+ " "+ str(row[10])+ " "+ str(row[12])))