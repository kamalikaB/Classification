import pandas as pd
import numpy as np
import os
import copy
import math
from numpy import random
from tabulate import tabulate
import warnings
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay)


class FDCA:
    def __init__(self, Dataset_name, split_size, num_class, comp_others =False,
                 trials=400, num_threads=1000) -> None:

        self.save_location = './temp/'
        self.trials = trials
        self.num_threads = num_threads
        self.labels_ = []
        self.best_so_far = 0
        self.Dataset_name = Dataset_name
        self.rule_kind = 'best'
        self.split_size = split_size
        self.num_class = num_class
        self.split_index = 0
        self.compare = comp_others
        warnings.filterwarnings("ignore")
        
        
    def main(paramString, cell_length):
        #****************This function find the cycleset of FDCA
        left = 1  # assuming left neighbors
        right = 1  # assuming right neighbors
        m = left + right + 1  # m is the number of neighboring cells
        cycle_set = []
        cycle = []
        try:
            # n = int(input("Enter the number of cells: "))
            n = cell_length
            # n=5
        except ValueError:
            print("Please enter a valid integer for the number of cells.")
            sys.exit(1)

        # Now you can proceed with the rest of your code logic
        d = 10  # Assuming default value for d
        
        # Example usage
        # print("Number of cells:", n)
        # print("State of the CA:", d)
        # print("Number of neighbors on each side:", left, "left and", right, "right")
        N = d ** n  # N = total no. of states of an n-cell CA
        noRMTs = d ** m  # Calculate the value of noRMTs using exponentiation
        # Initialize an array of integers
        Rule = [0] * noRMTs
        # Create an empty string
        ruleString = ""
        # paramString = input(f"Enter the parameters for {d} state radius 1 
        # first degree CA as comma separated values (c0xyz+c1xy+c2yz+c3zx+c4x+c5y+c6z+c7): ")
        # paramString = '0,0,0,0,0,1,0,1'
        words = paramString.split(",")
        param = [int(word.strip()) for word in words]  # parameters for first degree CA

        # Generate the rule
        Rule = []
        for x in range(d):
            for y in range(d):
                for z in range(d):
                    rule = (param[0]*x*y*z + param[1]*x*y + param[2]*x*z + param[3]*z*y + param[4]*x + param[5]*y + param[6]*z + param[7]) % d
                    Rule.append(rule)

        #print("The rule is: ")
        for i in range(len(Rule) - 1, -1, -1):
            #print(Rule[i], end="")
            ruleString += str(Rule[i])

        #print()
        check = [0] * N
        PS = [0] * n
        SS = [0] * n
        NS = [0] * n
        Comb = [0] * n
        
        
        flag = False
        check = [0] * N
        
        PS = [0] * n
        Comb = [0] * n
        
        
        #print("".join(map(str, PS)), end="")
        #print(" (0)")
        check[0] = 1

        while True:
            
            for i in range(n):
                SS[i] = PS[i]
            cycle=[]
            while True:
                # Next state generation and cycle checking
                num_str = ''.join(map(str, PS))
                PS_int=int(num_str)
                cycle.append(PS_int)
                #for i in range(n):
                    #print(PS[i], end="")
                 
                #print()
                
                

                for i in range(left):  # correct for leftmost cells
                    RMT = 0
                    range_val = m - left - 1 + i
                    
                    for j in range(m - left + i):
                        RMT += d ** range_val * PS[j]
                        range_val -= 1
                    
                    NS[i] = Rule[RMT]  # next state for ith cell
                    
                for i in range(left, n - right):  # correct for middle cells
                    # calculation of RMT
                    RMT = 0
                    range_val = m - 1
                    
                    for j in range(i - left, i + right + 1):
                        RMT += d ** range_val * PS[j]
                        range_val -= 1
                    
                    NS[i] = Rule[RMT]  # next state for ith cell

                for i in range(n - right, n):  # correct for rightmost cells
                    # calculation of RMT
                    RMT = 0
                    range_val = m - 1
                    
                for j in range(i - left, n):
                    RMT += d ** range_val * PS[j]
                    range_val -= 1
                
                NS[i] = Rule[RMT]  # next state for ith cell
                
                for i in range(n):
                    PS[i] = NS[i]

                index = 0
                for i in range(n):
                    index += PS[i] * d ** (n - i - 1)

                #print(" (" + str(index) + ")")
                
                if check[index] == 1:
                    #print(cycle)
                    cycle_set.append(cycle)
                    #for i in range(n):
                       #print(Comb[i], end="")
                    #print(" (" + str(index) + ")")
                    #print()
                    break
                else:
                    check[index] = 1
                
            while True:
                # Find an unexplored state
                c=0   
                for i in range(n):
                    if Comb[i] == (d - 1):
                        Comb[i] = 0
                        c=c+1
                    else:
                        break
                
                if c < n:
                    Comb[i] += 1
                else:
                    break
                #print(Comb)    
                index = 0
                for i in range(n):
                    index += Comb[i] * d ** (n - i - 1)
                #print(index)
                if check[index] == 0:
                    check[index] = 1
                    break
            c=0
    
            for i in range(n):
                if Comb[i] == 0:
                    c=c+1
                    continue
                else:
                    break

            if c == n:
                flag = True
                break

            for i in range(n):
                PS[i] = Comb[i]
        return cycle_set
    
    
    def generatePrime(n):
        X = 0
        i = 2
        flag = False
        PRIME = []
        while (X < n):
            flag = True
            for j in range(2, math.floor(math.sqrt(i)) + 1):
                if (i % j == 0):
                    flag = False
                    break
            if (flag):
                #print(i, end=" ")
                PRIME.append(i)
                X += 1
            i += 1
        #print()
        return PRIME
              
    
      
    def find_median(lst):
        # First, sort the list
        sorted_lst = sorted(lst)
        
        # Check if the length of the list is odd or even
        n = len(sorted_lst)
        if n % 2 == 0:
            # If the length is even, average the middle two elements
            middle_left = sorted_lst[n // 2 - 1]
            middle_right = sorted_lst[n // 2]
            median = (middle_left + middle_right) / 2
        else:
            # If the length is odd, return the middle element
            median = sorted_lst[n // 2]
        
        return median



    def check_integer_in_list_of_lists(integer, list_of_lists):
        for i, sublist in enumerate(list_of_lists):
            if integer in sublist:
                index_cycle = i
        return index_cycle


    def find_nearest_value(target, number_list):
        nearest_value = None
        min_difference = float('inf')

        for number in number_list:
            difference = abs(number - target)
            if difference < min_difference:
                min_difference = difference
                nearest_value = number

        return nearest_value

    def classification(cycle_set, train_data_configuration,train_data_class):
       #print(cycle_set,train_data_configuration)
        size = len(cycle_set)
        cycle_set_class = [0] * size
        # for ele in cycle_set:
        #     #print(ele)
        #     with open('output.txt', 'a') as file:
        #     # Write each element of the list to the file
        #         concatenated_str = ''.join(map(str, ele))
        #         file.write(str(concatenated_str) + '\n')
        #print(len(cycle_set),len(cycle_set_class))
        cl1 = 0
        cl2 = 0
        train_data_configuration_int=[]
        for train_data in train_data_configuration:
            concatenated_str = ''.join(map(str, train_data))
            concatenated_train_data = int(concatenated_str)
            train_data_configuration_int.append(concatenated_train_data)
        # finding class for all cycles
        
        for train_data in train_data_configuration:
            index_class_train = train_data_configuration.index(train_data)
            concatenated_str = ''.join(map(str, train_data))
            concatenated_train_data = int(concatenated_str)
            cl1 = 0
            cl2 = 0
            index_cycle = FDCA.check_integer_in_list_of_lists(concatenated_train_data, cycle_set)
            #print(f"The train data{concatenated_train_data} of class {train_data_class[index_class_train]} in cycle index {index_cycle} and cycle{cycle_set[index_cycle]}")
            if(cycle_set_class[index_cycle]==0):
                #cycle_set_class[index_cycle]=train_data_class[index_class_train]
                #print(concatenated_train_data , cycle_set[index_cycle])
                for ele in cycle_set[index_cycle]:
                    if ele in train_data_configuration_int:
                        #print(f"{ele} is present in the list.")
                        index_class_train = train_data_configuration.index(train_data)
                        clas=train_data_class[index_class_train]
                        if(clas == 1):
                          cl1=cl1+1
                        else:
                          cl2=cl2+1
                        #print(clas)
                if(cl1>cl2):
                  cycle_set_class[index_cycle]=1
                else:
                  cycle_set_class[index_cycle]=2
        # If cycle class is not find means no training dat information for that cycle.Then It is find by medain value and nearest training dataset value      
        #print("cycle set class",cycle_set_class)
        for index, element in enumerate(cycle_set_class):
            if(element == 0):
                #print(f"Index: {index}, Element: {element}")
                #print(cycle_set[index])
                median=FDCA.find_median(cycle_set[index])
                #print(median)
                nearest_train_data=FDCA.find_nearest_value(median,train_data_configuration_int)
                index_near=train_data_configuration_int.index(nearest_train_data)
                cycleclass=train_data_class[index_near]
                #print(f"Median: {median}, Train_data_nearest: {nearest_train_data} and class {cycleclass}")
                cycle_set_class[index]=cycleclass
                
                
        #for index, element in enumerate(cycle_set_class):
            #print(f"Index: {index}, Element: {element}")
        output_classification=[]
        output_classification.append(cycle_set)
        output_classification.append(cycle_set_class)
        #print("cycle set class",cycle_set_class)
        return output_classification
        


    def prediction(test_data_configuration,cycle_set,cycle_set_class):
        # print(test_data_configuration)
        y_pred_CA=[]
        test_data_configuration_int = []
        for test_data in test_data_configuration:
            concatenated_str = ''.join(map(str, test_data))
            concatenated_test_data = int(concatenated_str)
            test_data_configuration_int.append(concatenated_test_data)
        #print(test_data_configuration_int)
        for test_data in test_data_configuration_int:
            index_cycle = FDCA.check_integer_in_list_of_lists(test_data, cycle_set)
            clas=cycle_set_class[index_cycle]
            y_pred_CA.append(clas)
        return y_pred_CA

    def get_rule_list(self, path="./rule_list.txt"):
        # check if best rules exist
        best_path = './config/'+self.Dataset_name+'/Classification-' + \
            str(self.num_class)+'/'+self.rule_kind+'_rules.txt'
        try:
            print("\n********************FDCA************************")
            print("Found best configuration!")
            my_file = open(best_path, 'r')
            file_con = my_file.read()
            # replacing end splitting the text
            # when newline ('\n') is seen.
            rule_list = file_con.split("\n")
            my_file.close()
            self.num_threads = 1
            self.trials = 2
            #print(rule_list[0][0])
            rule_best = []
            rule_best.append(rule_list[0].zfill(8))
            return rule_best
        except:
            print("\n********************FDCA************************")
            print("No best rules yet. Running trials...")
        # opening the file in read mode
        my_file = open(path, "r")
        file_con = my_file.read()
        rule_list = file_con.split("\n")
        my_file.close()
        return rule_list

    def aggregate_scores(self):
        path = './'+self.save_location+self.Dataset_name + \
            '/Classification-'+str(self.num_class)+'/'
        main_df = pd.DataFrame()
        for files in os.listdir(path):
            if '.DS_Store' in files or '.csv' not in files or 'final_scores' in files:
                continue
            df = pd.read_csv(path+files, index_col=False)
            main_df = main_df.append(df)
            # L=df
            # with open("/content/result/result1.txt", 'a') as testwritefile:
            #  testwritefile.write(str(L))
            # testwritefile.write(str("\n"))
        main_df.columns = ["Index", "FDCA Rule1", "FDCA Rule2", "-",
                           "CA Silhoutte", "Heir Silhoutte", "Kmeans Silhoutte", "Birch Silhoutte"]
        main_df = main_df.sort_values('CA Silhoutte', ascending=False)
        main_df.reset_index(inplace=True, drop=True)
        #print(main_df)
        best_score = main_df['CA Silhoutte'].iloc[0]
        best_rules = [main_df['FDCA Rule1'].iloc[0],
                      main_df['FDCA Rule2'].iloc[0]]
        main_df.to_csv('./'+self.save_location+'/'+self.Dataset_name+'/Classification-' +
                       str(self.num_class)+'/'+self.
                       rule_kind+'_final_scores.csv')
        return best_score, best_rules

    def fit(self):

        os.makedirs('./'+self.save_location+self.Dataset_name+'/Classification-' +
                    str(self.num_class)+'/Final Clusters', exist_ok=True)
        os.makedirs('./config/'+self.Dataset_name+'/Classification-' +
                    str(self.num_class), exist_ok=True)
        self.data = pd.read_csv("./data/"+self.Dataset_name+".csv")
        #for monk1
        # d = pd.DataFrame(self.data)
        
        
        #For Monk2 is not balance
        
        print(self.data['Class'].value_counts())
        # from sklearn.utils import resample
        # data_majority=self.data[self.data.Class==0]
        # data_minority=self.data[self.data.Class==1]
        # data_minority_upsampled=resample(data_minority,
        #                                 replace=True,
        #                                 n_samples=290,
        #                                 random_state=123)
        # data_upsampled=pd.concat([data_majority,data_minority_upsampled])
        #print(data_upsampled.Class.value_counts())
   
        # # *******************Relacing class label to 1 and 2******************
        # d = pd.DataFrame(data_upsampled)
        
        
        #for Monk3
        from sklearn.utils import resample
        data_majority=self.data[self.data.Class==1]
        data_minority=self.data[self.data.Class==2]
        data_minority_upsampled=resample(data_minority,
                                        replace=True,
                                        n_samples=225,
                                        random_state=123)
        data_upsampled=pd.concat([data_majority,data_minority_upsampled])
        print(data_upsampled.Class.value_counts())
       
        d = pd.DataFrame(data_upsampled)
        
        #d.Class.replace((1,0), (1, 2), inplace=True)
        target = d.Class
        df = d
        df.pop('Class')
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df, target, test_size=0.2, random_state=44)
        
        #ML classical algorithm for comparison
        print("*************************MultinomialNB************************************")
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        classifier_NB=MultinomialNB()
        classifier_NB.fit(X_train,y_train)
        y_pred_NB = classifier_NB.predict(X_test)
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred_NB))
        print('Precision: %.3f' % precision_score(y_test,y_pred_NB))
        print('Recall: %.3f' % recall_score(y_test, y_pred_NB))
        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_NB))
        print('F1 Score: %.3f' % f1_score(y_test,y_pred_NB))

        print("**************************svm********************")
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        import matplotlib.pyplot as plt
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        svc = SVC(kernel='linear', C=10.0, random_state=1)
        svc.fit(X_train, y_train)
        y_pred_svm = svc.predict(X_test)
        print(classification_report(y_test, y_pred_svm))
        print('Precision: %.3f' % precision_score(y_test, y_pred_svm))
        print('Recall: %.3f' % recall_score(y_test, y_pred_svm))
        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_svm))
        print('F1 Score: %.3f' % f1_score(y_test, y_pred_svm))

        print("***************************DecisionTreeClassifier*********************")
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        y_predict_DT=clf.predict(X_test)
        accuracy_score(y_test,y_predict_DT)*100
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_predict_DT))
        from sklearn.metrics import classification_report
        print(classification_report(y_test,y_predict_DT))
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        print('Accuracy: %.3f' % accuracy_score(y_test, y_predict_DT))
        print('Precision: %.3f' % precision_score(y_test, y_predict_DT))
        print('Recall: %.3f' % recall_score(y_test, y_predict_DT))
        print('F1 Score: %.3f' % f1_score(y_test,y_predict_DT))


        print("*********************MLPClassifier*******************")
        from sklearn.neural_network import MLPClassifier
        clf1 = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
        y_predict_MLP=clf1.predict(X_test)
        accuracy_score(y_test,y_predict_MLP)*100
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_predict_MLP))
        from sklearn.metrics import classification_report
        print(classification_report(y_test,y_predict_MLP))
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        print('Accuracy: %.3f' % accuracy_score(y_test, y_predict_MLP))
        print('Precision: %.3f' % precision_score(y_test, y_predict_MLP))
        print('Recall: %.3f' % recall_score(y_test, y_predict_MLP))
        print('F1 Score: %.3f' % f1_score(y_test,y_predict_MLP))

        print("*************************KNeighborsClassifier***************")
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, y_train)
        y_predict_KNN=neigh.predict(X_test)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        print('Accuracy: %.3f' % accuracy_score(y_test, y_predict_KNN))
        print('Precision: %.3f' % precision_score(y_test, y_predict_KNN))
        print('Recall: %.3f' % recall_score(y_test, y_predict_KNN))
        print('F1 Score: %.3f' % f1_score(y_test,y_predict_KNN))
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_predict_KNN))
        from sklearn.metrics import classification_report
        print(classification_report(y_test,y_predict_KNN))


        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X_train, y_train)
        y_predict_LR=neigh.predict(X_test)
        from sklearn.metrics import classification_report
        print("**********************LinearRegression*********************")
        print(classification_report(y_test,y_predict_LR))
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        print('Accuracy: %.3f' % accuracy_score(y_test,y_predict_LR))
        print('Precision: %.3f' % precision_score(y_test, y_predict_LR))
        print('Recall: %.3f' % recall_score(y_test, y_predict_LR))
        print('F1 Score: %.3f' % f1_score(y_test,y_predict_LR))
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_predict_LR))





       
        
        #****************For concatenating
        
        
        df1=X_train.values.tolist()
        train_data_class=y_train.values.tolist()
        train_data_configuration=[]
        for x in df1:
                output=[]
                for e in x:
                    output.append(list(map(int, str(e))))
                output1=[]
                for e in output:
                    if len(e)==0:
                            e.insert(0, 0)
                    for e1 in e:
                        output1.append(e1)
                train_data_configuration.append(output1)

        
        df2=X_test.values.tolist()
        test_data_class=y_test.values.tolist()
        test_data_configuration=[]
        for x in df2:
                output=[]
                for e in x:
                    output.append(list(map(int, str(e))))
                output1=[]
                for e in output:
                    if len(e)==0:
                            e.insert(0, 0)
                    for e1 in e:
                        output1.append(e1)
                test_data_configuration.append(output1)
        
        maxLength = max(len(x) for x in train_data_configuration)
        #print(maxLength)
        train_data_configuration_padded=[]
        for ele in train_data_configuration:
            ele = ([0] * maxLength + ele)[-maxLength:]
            train_data_configuration_padded.append(ele)
        train_data_configuration = copy.deepcopy(train_data_configuration_padded)
        maxLength = max(len(x) for x in test_data_configuration)
        #print(maxLength)
        test_data_configuration_padded=[]
        for ele in test_data_configuration:
            ele = ([0] * maxLength + ele)[-maxLength:]
            test_data_configuration_padded.append(ele)
        test_data_configuration = copy.deepcopy(test_data_configuration_padded)
        
        if (len(train_data_configuration[0]) != len(test_data_configuration[0])) :
            if (len(train_data_configuration[0]) < len(test_data_configuration[0])):
                orginal_data_godal_padded = []
                maxLength = max(len(train_data_configuration[0]) , len(test_data_configuration[0]))
                for ele in train_data_configuration:
                    ele = ([0] * maxLength + ele)[-maxLength:]
                    orginal_data_godal_padded.append(ele)
                train_data_configuration = copy.deepcopy(orginal_data_godal_padded)
            else:
                orginal_data_godal_padded = []
                maxLength = max(len(train_data_configuration[0]) , len(test_data_configuration[0]))
                for ele in test_data_configuration:
                    ele = ([0] * maxLength + ele)[-maxLength:]
                    orginal_data_godal_padded.append(ele)
                test_data_configuration = copy.deepcopy(orginal_data_godal_padded)
       
        #print("Train data configurarion")
        # for ele in train_data_configuration:
        #     print(ele)
        # print("test_data_configuration")
        # for ele in test_data_configuration:
        #     print(ele)
        # if (len(train_data_configuration[0]) == len(test_data_configuration[0])) :
        #     print("***********equal****************")  
        #n1 = len(train_data_configuration[0])
        #print(n1)
        #paramString = '0,0,0,0,5,1,6,2'
        #cycle_set = FDCA.main(paramString, n1)
        rule_list_name = self.rule_kind+'_cycles_'+str(self.split_size)
        rulelist = self.get_rule_list('./rules/'+rule_list_name+'.txt')
        #print("bestrule", type(rulelist))
        CA = []
        for ele in rulelist:
            x = list(ele)
            CA.append(x)
        #print(CA)
        CA_classification(self, CA, train_data_configuration, train_data_class,
                          test_data_configuration, test_data_class, df2, 
                          y_test)


def R(c0, c1, c2, c3, c4, c5, c6, c7, x, y, z):
    return ((c0*x*y*z+c1*x*y+c2*x*z+c3*y*z+c4*x+c5*y+c6*z+c7) % 10)


def CA_classification(self, CA, train_data_configuration, train_data_class,
                      test_data_configuration, test_data_class, df2, y_test):
    """
    better_score_list = []
    best_CA_sill = -10000
    best_rule = []
    """
    output_data = []
    columns = ["Rule 1", "Rule 2", "Rule 3", "CA Silhoutte",
               "Heir Silhoutte", "Kmeans Silhoutte", "Birch Silhoutte"]
    #print(CA)
    
    for ca in range(0,len(CA)):
        FDCA_rule = "".join([str(i) for i in CA[ca]])
        print(CA[ca])
        n1 = len(train_data_configuration[0])
        #print("n1",n1)
        paramString = ','.join(map(str, CA[ca]))
        cycle_set = FDCA.main(paramString, n1)
        Median_cycles = []
        for cycle in cycle_set:
            median = FDCA.find_median(cycle)
            Median_cycles.append(median)
        #print(Median_cycles)
        output = []
        output.append(cycle_set)
        output.append(train_data_configuration)
        output.append(train_data_class)
        output.append(test_data_configuration)
        output.append(y_test)
        output_classification=FDCA.classification(output[0],output[1],output[2])
        # if(output_classification==0):
        #     Ca_accuracy=0
        # else:
        y_pred_CA=FDCA.prediction(output[3],output[0],output_classification[1]) #output[3] = testdata ououtput_classification[1]=cyclc set class
        y_pred_CA_np_array = np.array(y_pred_CA)
        from sklearn.metrics import accuracy_score
        Ca_accuracy = accuracy_score(output[4],  y_pred_CA_np_array) # output[4]=y_test
        #print(Ca_accuracy)
        from sklearn.metrics import classification_report
        print(classification_report(y_test,y_pred_CA_np_array))
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        print('Precision: %.3f' % precision_score(y_test,  y_pred_CA_np_array))
        print('Recall: %.3f' % recall_score(y_test, y_pred_CA_np_array))
        print('Accuracy: %.3f' % accuracy_score(y_test,  y_pred_CA_np_array))
        print('F1 Score: %.3f' % f1_score(y_test,  y_pred_CA_np_array))
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_pred_CA_np_array))
        #Ca_accuracy = accuracy_score(y_test,  y_pred_CA_np_array)
        # with open("./result.txt", 'a') as testwritefile:
        #     testwritefile.write(str(L))
        #     testwritefile.write(str("\n"))
        #print(classification_report(y_test,  y_pred_CA_np_array))
        # labels = [1, 2]
        # cm = confusion_matrix(y_test,  y_pred_CA_np_array, labels=labels)
        # disp = ConfusionMatrixDisplay(
        #     confusion_matrix=cm, display_labels=labels)
        # disp.plot()

        y_test.tolist()

        y_pred_CA_np_array.tolist()
        Heir_sill_new = 0
        Kmeans_sill_new = 0
        Birch_new = 0
        #print("Write", FDCA_rule)
        output_data.append([str(FDCA_rule), 0, 0, Ca_accuracy,
                           Heir_sill_new, Kmeans_sill_new, Birch_new])
        out_df = pd.DataFrame(data=output_data, columns=columns)
        out_df.to_csv('./'+self.save_location+'/'+self.Dataset_name+'/Classification-' +
                      str(self.num_class)+'/best_'+str(self.split_index)+'_tr_'+str(1)+'.csv')
        rule3 = 0
        best_score, best_rules = self.aggregate_scores()
        if self.best_so_far < best_score:
            self.best_so_far = best_score
            #print("Best silhoutte score :", self.best_so_far)
            # self.labels_ = enc_data_
            with open('./config/'+self.Dataset_name+'/Classification-'+str(self.num_class)+'/'+self.rule_kind+'_rules.txt', 'w') as file1:
                file1.write(str(best_rules[0]))

        if self.compare:
            my_data = [[self.best_so_far]]
            head = ["FDCA"]
            print(tabulate(my_data, headers=head, tablefmt="grid"))

        #print("---------------------***-----------------******----------------------***--------------------")


RCA = FDCA('Haber-man',
             9, 2, comp_others=True)
RCA.fit()
