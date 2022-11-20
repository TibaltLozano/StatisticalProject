import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as stats

from scipy.stats import beta as beta_dist
from scipy.stats import lognorm
import seaborn as sns


def pert_distribution(a,b,c,lamb):
    
    #Calculates core PERT statistics
    alpha = np.asarray(1 + (lamb * ((b-a) / (c-a))))
    beta = np.asarray(1 + (lamb * ((c-b) / (c-a))))
            
    mean = np.asarray((a + (lamb*b) + c) / (2+lamb))
    var = np.asarray(((mean-a) * (c-mean)) / (lamb+3))
    skew = np.asarray((2 * (beta - alpha) * np.sqrt(alpha + beta + 1)) / ((alpha + beta + 2) * np.sqrt(alpha * beta)))
    kurt = np.asarray(((lamb+2) * ((((alpha - beta)**2) * (alpha + beta + 1)
                ) + (alpha * beta * (alpha + beta + 2)))) / (alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)))
    range_val = np.asarray(c - a)

    median = (beta_dist(alpha, beta).median() * range_val) + a


    rvs_vals = (beta_dist(alpha, beta).rvs() * range_val) + a
    mu = np.asarray((a + (lamb*b) + c) / (2+lamb))
    sigma = np.asarray((c - a) / (2+lamb))
    std = np.asarray( beta_dist.std(alpha,beta))
  
    return mean,median,std,mu,sigma,rvs_vals


def getEventNumFromPERT(a,b,c,lamb):
     #Calculates core PERT statistics
    alpha = np.asarray(1 + (lamb * ((b-a) / (c-a))))
    beta = np.asarray(1 + (lamb * ((c-b) / (c-a))))    
    range_val = np.asarray(c - a)
    rvs_vals = (beta_dist(alpha, beta).rvs(size=1, random_state=None) * range_val) + a

    return rvs_vals


def lognormal_distribution(lb,ub):
    s = 0.954
    loc_value = lb
    scale_value = ub - lb
    x = np.linspace(lognorm.ppf(0.01, s,loc_value,scale_value), lognorm.ppf(0.99, s,loc_value,scale_value), 100)
    mean, var, skew, kurt = lognorm.stats(s, moments='mvsk',loc = loc_value,scale = scale_value)
    std = lognorm.std(s,loc = loc_value,scale = scale_value)
    median = lognorm.median(s,loc = loc_value, scale = scale_value)
    plt.plot(x, lognorm.pdf(x, s,loc_value,scale_value),'r-', lw=5, alpha=0.6, label='lognorm pdf')
    plt.show()
    
    print(f"Mean Value : {mean}\n")
    print(f"Media Value : {median}\n")
    print(f"std Value : {std}")


#Get Loss 
def getLossFromLogNormal(lb,ub):
    s = 0.954
    loc_value = lb
    scale_value = ub - lb
    r = lognorm.rvs(s,loc = loc_value,scale = scale_value)
    return r


#Read File
file = open("Data.csv","r")
reader = csv.reader(file)

items = []  # put the rows in csv to a list

name_list,prob_min_list,prob_most_list,prob_max_list,lb_loss_list,ub_loss_list = [],[],[],[],[],[]

#Header lists 
values_name_list = ["","prob_min","prob_most","prob_max","lb_loss","ub_loss"]

#insert data to each list
for row in reader:
    items.append(row)


#load each items
for i in range(1, len(items)):
    name_list.append(items[i][0])
    prob_min_list.append(items[i][1])
    prob_most_list.append(items[i][2])
    prob_max_list.append(items[i][3])
    lb_loss_list.append(items[i][4])
    ub_loss_list.append(items[i][5])
    
#Check the values are empty or invalid
for i in range (1,len(items) ):
    for j in range(1,6):
        if (items[i][j] == "" or items[i][j] == None):
            print(f"{name_list[i]} has empty cells: {values_name_list[j]} empty")
        if items[i][j].isdigit() == False:
            print(f"{name_list[i]} has invalid values: {values_name_list[j]} is invalid")

      
for i in range(len(items) - 1):
    if(int(prob_min_list[i]) < 0):
       print(f"{name_list[i]}'s prob_min must be same or bigger than 0")
    elif(int(prob_most_list[i]) < int(prob_min_list[i])):
      print(f"{name_list[i]}'s prob_most must be same or bigger than prob_min")   
    elif (int(prob_max_list[i]) < int(prob_most_list[i])):
      print(f"{name_list[i]}'s prob_max must be same or bigger than prob_most")  
    
    if(int(lb_loss_list[i]) < 0):
        print(f"{name_list[i]}'s lb_loss must be same or bigger than 0") 
    elif (int(lb_loss_list[i]) > int(ub_loss_list[i])):
        print(f"{name_list[i]}'s ub_loss must be same or bigger than lb_loss")            
    
    
    
    
    
        
''' Here we model the loss frequency with prob_min, prob_most ,prob_max as a pert distribution'''        
print("*" * 30)
print("\n")
print("Here we model the loss frequency with prob_min, prob_most ,prob_max as a pert distribution\n")

#contains the PERT minimum values in np.array.form
a = np.asarray([int(x) for x in prob_min_list])   

#contains the PERT maximum  values in np.array.form
c = np.asarray([int(x) for x in prob_max_list])

#contains the PERT most likely values in np.array.form
b = np.asarray([int(x) for x in prob_most_list])

lamb = 4.0 #The Pert Lambda parameter

mean,median,std,mu,sigma,rvs_vals = pert_distribution(a,b,c,lamb)

#plot the pert distribution result
res = sns.kdeplot(rvs_vals)
plt.title("Pert Distribution Result")
plt.show()

print(f"Mean Value : {mean}\n")
print(f"Median Value : {median}")
print(f"standard deviation:{std}")
print(f"mu Value : {mu}\n")
print(f"sigma Value : {sigma}")


           
''' Here we model the loss impact for each event with lb_loss and ub_loss as a lognormal distribution'''        
print("*" * 30)
print("\n")
print("Here we model the loss impact for each event with lb_loss and ub_loss as a lognormal distribution\n")

lb_list = [int(x) for x in lb_loss_list]
ub_list = [int(x) for x in ub_loss_list]


for i in range(len(lb_list)):
    plt.title(f"Log Normal Distribution probability density function in {name_list[i]}")
    lognormal_distribution(lb_list[i],ub_list[i])  
    
'''Here we simulate the simple Monte Carlo Simulation '''
simulation_steps = 5000


print("Monte Carlo Simulation Starts...\n")
for  year  in range(simulation_steps):
    print(f"Step {year}:\n")
    events_list = []
    loss_list = []
    total_loss = 0
    for j in range(len(prob_min_list)):
        events_num = getEventNumFromPERT(a[j],b[j],c[j],lamb=4)
        events_list.append(int(events_num[0]))
        for p in range(len(lb_list)):
            loss_num = getLossFromLogNormal(lb_list[p],ub_list[p])
            loss_list.append(int(loss_num))
            total_loss += loss_num
    '''Here we calculated some features'''
    average_event_per_year = int(np.mean(events_list))
    print(f"Average Event number : {average_event_per_year}\n")
    min_num_event = np.min(events_list)
    print(f"Min number of Events: {min_num_event}\n")
    max_num_event = np.max(events_list)
    print(f"Max number of Events: {max_num_event}\n")

    average_loss_per_year = int(np.mean(loss_list))
    print(f"Average Loss : {average_loss_per_year}\n")
    min_loss = np.min(loss_list)
    print(f"Min Loss: {min_loss}\n")
    max_loss = np.max(loss_list)
    print(f"Max Loss: {max_loss}\n")      
    
      






