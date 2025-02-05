# -*- coding: utf-8 -*-
"""ACRI_Extention_Version5_Hypo_encoding_train_test_separate_cross validation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DBBj_TBDLxYQRbqpHM0t_iWE1Xy5CSik
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('./data/Glass.csv')

n=5

df.head()

# df['A1'] = df['A1'].astype('object')
# df['A2'] = df['A2'].astype('object')
# df['A3'] = df['A3'].astype('object')
# df['A4'] = df['A4'].astype('object')

#print(df['Class'].value_counts())

# from sklearn.utils import resample
# data_majority=df[df.Class=='g']
# data_minority=df[df.Class=='b']
# data_minority_upsampled=resample(data_minority,
#                                 replace=True,
#                                 n_samples=225,
#                                 random_state=123)
# data_upsampled=pd.concat([data_majority,data_minority_upsampled])
# print(data_upsampled.Class.value_counts())

# df = pd.DataFrame(data_upsampled)

# Define the Preprocessor class
class Preprocessor:
    def __init__(self):
        self.label_encoders = {}  # To store label encoders for categorical columns
        self.bin_intervals = {}  # To store bin intervals for numerical columns

    def preprocess_training_data(self, X_train, y_train):
        # Process categorical columns

        for column in X_train.select_dtypes(include=[np.number]).columns:
            bins = np.linspace(X_train[column].min(), X_train[column].max(), 10)
            X_train[column] = np.digitize(X_train[column], bins) - 1
            self.bin_intervals[column] = bins  # Store the bins



        for column in X_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_train[column] = le.fit_transform(X_train[column])
            #print(X_train[column])
            self.label_encoders[column] = le  # Store the encoder

        # Process numerical columns

        # Encode target column
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        self.label_encoders['Class'] = le  # Store the encoder for the target column

        return X_train, y_train

    def preprocess_test_data(self, X_test, y_test):
        #print(self.bin_intervals)
         # Process numerical columns
        for column in X_test.select_dtypes(include=[np.number]).columns:
            if column in self.bin_intervals:
                bins = self.bin_intervals[column]
                X_test[column] = np.digitize(X_test[column], bins) - 1
                if X_test[column].min() < 0:
            # Find the closest bin interval for negative values
                   X_test[column] = np.clip(X_test[column], 0, len(bins) - 2)
            else:
                raise ValueError(f"Bin intervals for column '{column}' not found.")

        # Process categorical columns
        for column in X_test.select_dtypes(include=['object']).columns:
            if column in self.label_encoders:
                le = self.label_encoders[column]
                X_test[column] = le.transform(X_test[column])
                if X_test[column].min() < 0:
                    X_test[column] += 1
            else:
                raise ValueError(f"Label encoder for column '{column}' not found.")



        # Encode target column
        if 'Class' in self.label_encoders:
            le = self.label_encoders['Class']
            y_test = le.transform(y_test)
        else:
            raise ValueError("Label encoder for 'Class' not found.")

        return X_test, y_test

def traintestsplit(df, random_state_i,test_size=0.2):
  import pandas as pd
  import numpy as np
  from sklearn.preprocessing import LabelEncoder
  from sklearn.model_selection import train_test_split



  # # Sample DataFrame (replace with your actual data)
  # data = {
  #     'A': ['X', 'Y', 'Z', 'X', 'Y'],
  #     'B': [1.5, 2.3, 3.1, 4.2, 5.0],
  #     'Class': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
  # }

  # # Create DataFrame
  # df = pd.DataFrame(data)
  y = df['Class']
  # Split data into features and target
  df=df.drop(columns=['Class'])
  X = df


  # Split data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_i)

  # Initialize the Preprocessor
  preprocessor = Preprocessor()

  # Preprocess training data
  X_train, y_train = preprocessor.preprocess_training_data(X_train, y_train)

  # Preprocess test data using the same preprocessor
  X_test, y_test = preprocessor.preprocess_test_data(X_test, y_test)

  # Concatenate all the columns in training data into a single column
  X_train['Concatenated'] = X_train.astype(str).agg(''.join, axis=1)
  #print(X_train)
  # Concatenate all the columns in test data into a single column
  X_test['Concatenated'] = X_test.astype(str).agg(''.join, axis=1)
  #print(X_test)

  # Padding function to ensure concatenated strings are multiples of n
  def pad_to_multiple_of_n(s):
      while len(s) % n != 0:
          s = '0' + s
      return s

  # Apply padding to the concatenated column
  X_train['Concatenated'] = X_train['Concatenated'].apply(pad_to_multiple_of_n)
  X_test['Concatenated'] = X_test['Concatenated'].apply(pad_to_multiple_of_n)

  # Split concatenated strings in training data into n-length substrings
  X_train_split = []
  y_train_split = []

  for concatenated, class_label in zip(X_train['Concatenated'], y_train):
      split_values = [concatenated[i:i+n] for i in range(0, len(concatenated), n)]
      X_train_split.extend(split_values)
      y_train_split.extend([class_label] * len(split_values))

  # Split concatenated strings in test data into n-length substrings
  X_test_split = []
  y_test_split = []

  for concatenated, class_label in zip(X_test['Concatenated'], y_test):
      split_values = [concatenated[i:i+n] for i in range(0, len(concatenated), n)]
      X_test_split.extend(split_values)
      y_test_split.extend([class_label] * len(split_values))

  # Convert splits into DataFrames
  X_train_split_df = pd.DataFrame(X_train_split, columns=['Attribute'])
  y_train_split_df = pd.Series(y_train_split, name='Class')

  X_test_split_df = pd.DataFrame(X_test_split, columns=['Attribute'])
  y_test_split_df = pd.Series(y_test_split, name='Class')

  # Display the split DataFrames
  #print("Training Data Split:")
  #print(X_train_split_df)
  #print(y_train_split_df)

  #print("\nTest Data Split:")
  #print(X_test_split_df)
  #print(y_test_split_df)
  train_data_configuration=X_train_split_df.values.tolist()
  train_data_configuration = [list(item[0].ljust(n, '0')) for item in train_data_configuration]
  train_data_class=y_train_split_df.values.tolist()
  return train_data_configuration, train_data_class,X_test,y_test



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

#0,0,0,0,2,1,6,3
#CA=['0', '0', '0', '0', '2', '1', '6', '3']


# FDCA_rule = "".join([str(i) for i in CA])
# #print(CA[ca])

# #print("n1",n1)
# paramString = ','.join(map(str, CA))
# cycle_set = main(paramString, 4)

def get_rule_list(path):
  my_file = open(path, "r")
  file_con = my_file.read()
  rule_list = file_con.split("\n")
  return rule_list

rulelist = get_rule_list('./small_datarules_52.txt')
#print("bestrule", type(rulelist))
CA = []
for ele in rulelist:
    x = list(ele)
    CA.append(x)

#print(CA)

def CA_classification(ca, train_data_configuration, train_data_class,X_test):
    FDCA_rule = "".join([str(i) for i in ca])
    #print(CA[ca])
    n1 = len(train_data_configuration[0])
    #print("n1",n1)
    paramString = ','.join(map(str, ca))
    cycle_set = main(paramString, n1)
    #print(len(cycle_set))
    # Pad each element in each list of cycle_set to ensure it's of length n
    # Format padded_cycle_set to ensure each element has length n
    padded_cycle_set = [[str(element).zfill(n) for element in sublist] for sublist in cycle_set]


    # Format train_data_configuration to join each sublist into a single string and pad to length n
    train_data_configuration = [''.join(sublist).zfill(n) for sublist in train_data_configuration]
    #print(train_data_configuration)
    # Display the results
    # print(padded_cycle_set)
    # print(train_data_configuration)
    # print(len(train_data_configuration))
    #print(train_data_class)
    # print(len(train_data_class))
    from collections import Counter

    # Example setup (replace these with your actual data)
    # padded_cycle_set = [['00000', '11111', '22222', ...], ...]
    # train_data_configuration = ['00000', '11111', '22222', ...]
    # train_data_class = [1, 0, 1, ...]  # Corresponding class labels for each element in train_data_configuration

    # Create a dictionary mapping each element in train_data_configuration to its class
    train_data_dict = dict(zip(train_data_configuration, train_data_class))

    # Initialize a list to store the class label for each cycle in padded_cycle_set
    cycle_labels = []


    for cycle in padded_cycle_set:
        #print(cycle)
        # Initialize a list to store the class labels for elements in the current cycle
        # Get the class labels for elements in the current cycle and print the element
        element_classes = []
        for element in cycle:
            if element in train_data_dict:
                element_classes.append(train_data_dict.get(element))
                #print(f"Element: {element}, Class: {train_data_dict.get(element)}")

            # Determine the majority class if there are elements with known classes
            if element_classes:
                #print(cycle)
                # Find the most common class label
                majority_class = Counter(element_classes).most_common(1)[0][0]
                #print(majority_class)
            else:
                # Assign a default label (e.g., -1) if no known classes are found
                majority_class = -1

            # Append the majority class label for the cycle
        #print(majority_class)
        cycle_labels.append(majority_class)
    #print(cycle_labels)

    import numpy as np
    from collections import Counter

    # Example data (replace with your actual data)
    # padded_cycle_set = [['00000', '11111', '22222', ...], ...]  # List of cycles
    # cycle_labels = [1, -1, 0, -1, 1, ...]  # Corresponding list of class labels, some are -1

    # Step 1: Calculate the median of each cycle's elements
    cycle_medians = []

    # Calculate the median for each cycle
    for cycle in padded_cycle_set:
        # Convert cycle elements to numeric values (if necessary) and compute the median
        cycle_elements = [int(element) for element in cycle]  # Assuming elements are strings that can be converted to integers
        median_value = np.median(cycle_elements)
        cycle_medians.append(median_value)

    # Step 2: Find the cycles with label -1
    for i, cycle_label in enumerate(cycle_labels):
        if cycle_label == -1:
            # Step 3: Find the cycle whose median is closest to the current cycle's median
            closest_distance = float('inf')
            closest_cycle_label = None

            for j, other_median in enumerate(cycle_medians):
                if cycle_labels[j] != -1:  # Skip cycles that are already labeled
                    distance = abs(cycle_medians[i] - other_median)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_cycle_label = cycle_labels[j]

            # Step 4: Update the cycle label with the closest cycle's label
            cycle_labels[i] = closest_cycle_label

    # Display the updated cycle labels
    #print(padded_cycle_set)
    #print(cycle_labels)
    import numpy as np
    from collections import Counter

    # Assuming X_test is a DataFrame with a column 'Concatenated' containing the strings to classify
    # padded_cycle_set is a list of lists of strings, e.g., [['00000', '11111', '22222', ...], ...]
    # cycle_labels corresponds to the labels of each cycle in padded_cycle_set.


    majority_labels = []
    #print(X_test)
    # Step 1: Iterate over each row in X_test
    for idx, row in X_test.iterrows():
        test_string = row['Concatenated']
        #print(test_string)
        # Step 2: Split the string into substrings of length n
        substrings = [test_string[i:i+n] for i in range(0, len(test_string), n)]
        #print(substrings)
        # Step 3: Initialize a list to hold the labels corresponding to the substrings
        labels = []

        # Step 4: For each substring, check if it's in any of the cycles in padded_cycle_set
        for substring in substrings:
            for cycle_idx, cycle in enumerate(padded_cycle_set):
                if substring in cycle:
                    # Get the label of the corresponding cycle
                    labels.append(cycle_labels[cycle_idx])
                    break  # Since we found a match, no need to check further cycles for this substring

        # Step 5: Find the majority class of the labels for this row
        if labels:
            majority_class = Counter(labels).most_common(1)[0][0]
        else:
            majority_class = -1  # In case there are no valid matches, set to -1 or other placeholder

        majority_labels.append(majority_class)

    # Output the predicted labels for X_test
    #print(majority_labels)

    y_pred_CA_np_array = np.array(majority_labels)
    return y_pred_CA_np_array
    # from sklearn.metrics import accuracy_score
    # Ca_accuracy = accuracy_score(y_test,  y_pred_CA_np_array)
    # print(CA[ca],Ca_accuracy)

df

CA



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
#CA=[['0', '0', '0', '0', '4', '1', '0', '8']] #n=4 Breast Cancer
CA=[['0', '0', '0', '0', '7', '9', '0', '8']] #n=5 Glass

for ca in range(0,len(CA)):
  #print(CA[ca])
  accuracies_ca = []  # List to store accuracy values
  n_repeats = 10
  for i in range(n_repeats):
      #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
      train_data_configuration, train_data_class,X_test,y_test=traintestsplit(df,i)
      y_pred_CA_np_array=CA_classification(CA[ca], train_data_configuration, train_data_class,X_test)
      from sklearn.metrics import accuracy_score
      import numpy as np

      # Assuming 'accuracies' and 'y_test', 'y_pred_CA_np_array' are defined previously

      # Calculate accuracy for each fold and append to the list
      acc = accuracy_score(y_test, y_pred_CA_np_array)
      accuracies_ca.append(acc)
      print(acc)

      # Calculate mean accuracy and standard deviation
  #print(accuracies)
  mean_accuracy = np.mean(accuracies_ca)
  std_dev = np.std(accuracies_ca)

  # Assuming 'CA' and 'ca' are defined earlier
  result_ca = f"{mean_accuracy*100:.2f} ± {std_dev*100:.2f} {CA[ca]}"  # Format as required

  # Print the result to console
  print(result_ca)
  #print(accuracies_ca)
  # Write the result to a file (append mode, so results won't be overwritten)
  with open("./results/Glass_5", "a") as file:
      file.write(result_ca + "\n")  # Add newline to separate results

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('./data/Glass.csv')
# Split the DataFrame into features (X) and target (y)
X = df.drop(columns=['Class'])  # Replace 'target' with your actual target column name
y = df['Class']

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_SVM = []

# Assuming 'df' is your DataFrame, and the target column is 'target'


# Number of repeats (e.g., 10)
n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
for i in range(n_repeats):
    # Train-test split for each repeat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the SVM classifier
    model = SVC(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
    acc = accuracy_score(y_test, y_pred)
    accuracies_SVM.append(acc)
#print(accuracies)
# Calculate mean accuracy and standard deviation
mean_accuracy = np.mean(accuracies_SVM)
std_dev = np.std(accuracies_SVM)

# Format the result as percentage with two decimal places
result_svm = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"
print(result_svm)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_KNN = []

# Assuming 'df' is your DataFrame, and the target column is 'target'
# Split the DataFrame into features (X) and target (y)
# X = df.drop(columns=['target'])  # Replace 'target' with your actual target column name
# y = df['target']

# Number of repeats (e.g., 10)
n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
for i in range(n_repeats):
    # Train-test split for each repeat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the KNN classifier (you can adjust 'n_neighbors' based on your data)
    model = KNeighborsClassifier(n_neighbors=5)  # Example with k=5 neighbors

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
    acc = accuracy_score(y_test, y_pred)
    accuracies_KNN.append(acc)
#print(accuracies)
# Calculate mean accuracy and standard deviation
mean_accuracy = np.mean(accuracies_KNN)
std_dev = np.std(accuracies_KNN)

# Format the result as percentage with two decimal places
result_KNN = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"  # Format as required

# Print the result to console
print(result_KNN)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_RF = []

# Number of repeats (e.g., 10)
n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
for i in range(n_repeats):
    # Train-test split for each repeat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the Random Forest classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)  # You can tune n_estimators as needed

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
    acc = accuracy_score(y_test, y_pred)
    accuracies_RF.append(acc)

# Calculate mean accuracy and standard deviation
mean_accuracy = np.mean(accuracies_RF)
std_dev = np.std(accuracies_RF)

# Format the result as percentage with two decimal places
result_RF = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"
print(result_RF)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_DT = []

# Assuming 'X' is your feature matrix and 'y' is your target variable
# Number of repeats (e.g., 10)
n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
for i in range(n_repeats):
    # Train-test split for each repeat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the Decision Tree classifier
    model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
    acc = accuracy_score(y_test, y_pred)
    accuracies_DT.append(acc)

# Calculate mean accuracy and standard deviation
mean_accuracy = np.mean(accuracies_DT)
std_dev = np.std(accuracies_DT)

# Format the result as percentage with two decimal places
result_DT = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"

# Print the result
print(result_DT)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_MLP = []

# Assuming 'X' is your feature matrix and 'y' is your target variable
# Number of repeats (e.g., 10)
n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
for i in range(n_repeats):
    # Train-test split for each repeat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the MLP classifier (you can adjust parameters such as hidden_layer_sizes)
    model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)  # Example with one hidden layer of 100 neurons

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
    acc = accuracy_score(y_test, y_pred)
    accuracies_MLP.append(acc)

# Calculate mean accuracy and standard deviation
mean_accuracy = np.mean(accuracies_MLP)
std_dev = np.std(accuracies_MLP)

# Format the result as percentage with two decimal places
result_MLP = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"

# Print the result
print(result_MLP)

#& SVM & KNN &RF & DT & MLP &  ALRFRS & Decimal CA \\

print("SVM:",result_svm)
print("KNN:",result_KNN)
print("RF:",result_RF)
print("DT:",result_DT)
print("MLP:",result_MLP)
#print("ALRFRS:",result_ALRFRS)
print("CA:",result_ca)







for ca in range(0,len(CA)):
  #print(CA[ca])
  accuracies_ca = []  # List to store accuracy values
  #n_repeats = 1
  i=5
  #for i in range(n_repeats):
      #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
  train_data_configuration, train_data_class,X_test,y_test=traintestsplit(df,i)
  y_pred_CA_np_array=CA_classification(CA[ca], train_data_configuration, train_data_class,X_test)
  from sklearn.metrics import accuracy_score
  import numpy as np

    # Assuming 'accuracies' and 'y_test', 'y_pred_CA_np_array' are defined previously

    # Calculate accuracy for each fold and append to the list
  acc_ca = accuracy_score(y_test, y_pred_CA_np_array)
  # accuracies_ca.append(acc)
  print(acc_ca)
      # Calculate mean accuracy and standard deviation
  #print(accuracies_ca)
  #mean_accuracy = np.mean(accuracies_ca)
  #std_dev = np.std(accuracies_ca)

  # Assuming 'CA' and 'ca' are defined earlier
  #result_ca = f"{mean_accuracy*100:.2f} ± {std_dev*100:.2f} {CA[ca]}"  # Format as required

  # Print the result to console
  #print(result_ca)
  #print(accuracies_ca)
  # Write the result to a file (append mode, so results won't be overwritten)
#   with open("./results/results.txt", "a") as file:
#       file.write(result_ca + "\n")  # Add newline to separate results

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('./data/Haber-man.csv')

# Split the DataFrame into features (X) and target (y)
X = df.drop(columns=['Class'])  # Replace 'Class' with your actual target column name
y = df['Class']

# Encode the target column 'y' (Class) using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Encode categorical features in X using LabelEncoder
# Apply LabelEncoder to each column in X that contains strings
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

# List to store accuracy values
accuracies_SVM = []

# Number of repeats (e.g., 10)
#n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
#for i in range(n_repeats):
    # Train-test split for each repeat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the SVM classifier
model = SVC(random_state=42)

    # Train the model
model.fit(X_train, y_train)

    # Predict on the test set
y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
acc_swm = accuracy_score(y_test, y_pred)
#accuracies_SVM.append(acc)

# Calculate mean accuracy and standard deviation
#mean_accuracy = np.mean(accuracies_SVM)
#std_dev = np.std(accuracies_SVM)

# Format the result as percentage with two decimal places
#result_svm = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"
#print(result_svm)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_KNN = []

# Assuming 'df' is your DataFrame, and the target column is 'target'
# Split the DataFrame into features (X) and target (y)
# X = df.drop(columns=['target'])  # Replace 'target' with your actual target column name
# y = df['target']

# Number of repeats (e.g., 10)
#n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
#for i in range(n_repeats):
# Train-test split for each repeat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the KNN classifier (you can adjust 'n_neighbors' based on your data)
model = KNeighborsClassifier(n_neighbors=5)  # Example with k=5 neighbors

    # Train the model
model.fit(X_train, y_train)

    # Predict on the test set
y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
acc_KNN = accuracy_score(y_test, y_pred)
#accuracies_KNN.append(acc)
#print(accuracies)
# Calculate mean accuracy and standard deviation
#mean_accuracy = np.mean(accuracies_KNN)
#std_dev = np.std(accuracies_KNN)

# Format the result as percentage with two decimal places
#result_KNN = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"  # Format as required

# Print the result to console
#print(result_KNN)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_RF = []

# Number of repeats (e.g., 10)
#n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
#for i in range(n_repeats):
# Train-test split for each repeat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

# Initialize the Random Forest classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)  # You can tune n_estimators as needed

    # Train the model
model.fit(X_train, y_train)

    # Predict on the test set
y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
acc_RF = accuracy_score(y_test, y_pred)
#accuracies_RF.append(acc)

# Calculate mean accuracy and standard deviation
#mean_accuracy = np.mean(accuracies_RF)
#std_dev = np.std(accuracies_RF)

# Format the result as percentage with two decimal places
#result_RF = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"
#print(result_RF)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_DT = []

# Assuming 'X' is your feature matrix and 'y' is your target variable
# Number of repeats (e.g., 10)
#n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
#for i in range(n_repeats):
    # Train-test split for each repeat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)

    # Train the model
model.fit(X_train, y_train)

    # Predict on the test set
y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
acc_DT = accuracy_score(y_test, y_pred)
accuracies_DT.append(acc)

# Calculate mean accuracy and standard deviation
#mean_accuracy = np.mean(accuracies_DT)
#std_dev = np.std(accuracies_DT)

# Format the result as percentage with two decimal places
#result_DT = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"

# Print the result
#print(result_DT)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# List to store accuracy values
accuracies_MLP = []

# Assuming 'X' is your feature matrix and 'y' is your target variable
# Number of repeats (e.g., 10)
#n_repeats = 10

# Loop over n_repeats to perform the classification multiple times
#for i in range(n_repeats):
    # Train-test split for each repeat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize the MLP classifier (you can adjust parameters such as hidden_layer_sizes)
model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)  # Example with one hidden layer of 100 neurons

    # Train the model
model.fit(X_train, y_train)

    # Predict on the test set
y_pred = model.predict(X_test)

    # Calculate accuracy for this fold and append to the list
acc_MLP = accuracy_score(y_test, y_pred)
accuracies_MLP.append(acc)

# Calculate mean accuracy and standard deviation
#mean_accuracy = np.mean(accuracies_MLP)
#std_dev = np.std(accuracies_MLP)

# Format the result as percentage with two decimal places
#result_MLP = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"

# Print the result
#print(result_MLP)

#& SVM & KNN &RF & DT & MLP &  ALRFRS & Decimal CA \\

print("SVM:",acc_swm)
print("KNN:",acc_KNN)
print("RF:",acc_RF)
print("DT:",acc_DT)
print("MLP:",acc_MLP)
#print("ALRFRS:",result_ALRFRS)
print("CA:",acc_ca)











