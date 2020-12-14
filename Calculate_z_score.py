from igraph import *
import math
import csv

#resd the network
graph=Graph.Read_Ncol("Try.txt",names=True)



print("maxdegree")
print(graph.maxdegree())

# get degree centrality
degree_list = graph.vs.degree()

# get closeness centrality
close = graph.closeness(normalized= False)

#geet eigenvector centreality
evcentrality = graph.evcent(directed=False)

#get betweenness centrality
between = graph.betweenness(directed=False)
count = 0

#get average
avg_list=[]
for v in graph.vs:
    #print(v["name"] + " " +str(degree_list[count]) + " " + str(close[count]) + " " + str(between[count]) + " " + str(evcentrality[count]))
    avg= (degree_list[count]+close[count]+between[count]+evcentrality[count])/4.0
    avg_list.append(avg)
    count = count + 1
    #print()
tot =0.0
for i in range(0, len(avg_list)):
    tot = tot + avg_list[i]
avg_tot = tot/len(avg_list)
tot = 0.0

# get standard deviation
for i in range(0, len(avg_list)):
    tot += (avg_list[i]-avg_tot)*(avg_list[i]-avg_tot)
tot = tot/len(avg_list)
std_dev = math.sqrt(tot)
total_hub = 0

# calculate the z score
z_score = []
for i in range(0, len(avg_list)):
    z_score.append((avg_list[i] - avg_tot)/std_dev)
for i in range(0, len(z_score)):
    if z_score[i] >= 1:
        total_hub += 1
print(total_hub)
print(len(z_score))

# print the result in a graph
count = 0
avg =0.0
with open('Bucket2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Degree", "Closeness", "Betweenness", "Evcent", "Avg", "z-score"])
    avg = 0.0
    for v in graph.vs:
        avg = (degree_list[count]+close[count]+between[count]+evcentrality[count])/4.0
        writer.writerow([v["name"], degree_list[count], close[count], between[count], evcentrality[count], avg, z_score[count]])
        count = count + 1
count = 0
for v in graph.vs:
    print(v["name"] + " " +str(degree_list[count]) + " " + str(close[count]) + " " + str(between[count]) + " " + str(evcentrality[count]))
    count = count + 1
    print()
