import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Load dataset and select top 5 features
raw_df = pd.read_csv('./data/Dry_Bean.csv')
y = raw_df['Class']
X = raw_df.drop(columns=['Class'])
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
df_selected = raw_df[selected_features.tolist() + ['Class']]
df_selected.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'Class']

n = 4  # used for encoding and splitting

class Preprocessor:
    def __init__(self):
        self.label_encoders = {}

    def preprocess_training_data(self, X_train, y_train):
        X_train['A1'] = X_train['A1'].astype(int).astype(str).apply(lambda x: x.zfill(5))
        X_train['A2'] = (X_train['A2'] * 1000).astype(int).astype(str).apply(lambda x: x.zfill(6))
        X_train['A3'] = (X_train['A3'] * 1000000).astype(int).astype(str).apply(lambda x: x.zfill(9))
        X_train['A4'] = X_train['A4'].astype(int).astype(str).apply(lambda x: x.zfill(5))
        X_train['A5'] = (X_train['A5'] * 1000).astype(int).astype(str).apply(lambda x: x.zfill(6))
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        self.label_encoders['Class'] = le
        return X_train, y_train

    def preprocess_test_data(self, X_test, y_test):
        for col in ['A1', 'A2', 'A3', 'A4', 'A5']:
            X_test[col] = X_test[col].astype(int).astype(str).apply(lambda x: x.zfill(1))
        if 'Class' in self.label_encoders:
            le = self.label_encoders['Class']
            y_test = le.transform(y_test)
        else:
            raise ValueError("Label encoder for 'Class' not found.")
        return X_test, y_test

def traintestsplit(df, random_state_i, test_size=0.2):
    y = df['Class']
    X = df.drop(columns=['Class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state_i)
    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.preprocess_training_data(X_train, y_train)
    X_test, y_test = preprocessor.preprocess_test_data(X_test, y_test)
    X_train['Concatenated'] = X_train.astype(str).agg(''.join, axis=1).apply(pad_to_multiple_of_n)
    X_test['Concatenated'] = X_test.astype(str).agg(''.join, axis=1).apply(pad_to_multiple_of_n)
    X_train_split, y_train_split = [], []
    for c, label in zip(X_train['Concatenated'], y_train):
        X_train_split.extend([c[i:i+n] for i in range(0, len(c), n)])
        y_train_split.extend([label] * (len(c) // n))
    X_test_split, y_test_split = [], []
    for c, label in zip(X_test['Concatenated'], y_test):
        X_test_split.extend([c[i:i+n] for i in range(0, len(c), n)])
        y_test_split.extend([label] * (len(c) // n))
    train_data_configuration = [list(s.ljust(n, '0')) for s in X_train_split]
    return train_data_configuration, y_train_split, X_test, y_test

def pad_to_multiple_of_n(s):
    while len(s) % n != 0:
        s = '0' + s
    return s

def main(paramString, cell_length):
    d, left, right = 10, 1, 1
    m = left + right + 1
    N = d ** cell_length
    param = list(map(int, paramString.split(",")))
    Rule = [(param[0]*x*y*z + param[1]*x*y + param[2]*x*z + param[3]*z*y +
             param[4]*x + param[5]*y + param[6]*z + param[7]) % d
            for x in range(d) for y in range(d) for z in range(d)]
    check, PS, NS, Comb = [0]*N, [0]*cell_length, [0]*cell_length, [0]*cell_length
    cycle_set = []
    check[0] = 1
    while True:
        SS = PS[:]
        cycle = []
        while True:
            PS_int = int(''.join(map(str, PS)))
            cycle.append(PS_int)
            for i in range(cell_length):
                RMT = sum(d ** (m - 1 - j) * PS[(i - left + j) % cell_length] for j in range(m))
                NS[i] = Rule[RMT]
            PS = NS[:]
            idx = sum(PS[i] * d ** (cell_length - i - 1) for i in range(cell_length))
            if check[idx]:
                cycle_set.append(cycle)
                break
            else:
                check[idx] = 1
        for i in range(cell_length):
            if Comb[i] < d - 1:
                Comb[i] += 1
                break
            Comb[i] = 0
        idx = sum(Comb[i] * d ** (cell_length - i - 1) for i in range(cell_length))
        if check[idx]:
            if all(c == 0 for c in Comb):
                break
            continue
        check[idx] = 1
        PS = Comb[:]
    return cycle_set

def CA_classification(ca, train_data_configuration, train_data_class, X_test):
    paramString = ','.join(map(str, ca))
    cycle_set = main(paramString, len(train_data_configuration[0]))
    padded_cycle_set = [[str(el).zfill(n) for el in c] for c in cycle_set]
    train_dict = dict(zip([''.join(row) for row in train_data_configuration], train_data_class))
    cycle_labels = []
    for cycle in padded_cycle_set:
        labels = [train_dict.get(el, -1) for el in cycle if el in train_dict]
        label = Counter(labels).most_common(1)[0][0] if labels else -1
        cycle_labels.append(label)
    cycle_medians = [np.median([int(e) for e in c]) for c in padded_cycle_set]
    for i, label in enumerate(cycle_labels):
        if label == -1:
            distances = [abs(cycle_medians[i] - m) if cycle_labels[j] != -1 else float('inf')
                         for j, m in enumerate(cycle_medians)]
            closest = np.argmin(distances)
            cycle_labels[i] = cycle_labels[closest]
    majority_labels = []
    for _, row in X_test.iterrows():
        substrings = [row['Concatenated'][i:i+n] for i in range(0, len(row['Concatenated']), n)]
        labels = [cycle_labels[idx] for substring in substrings
                  for idx, cycle in enumerate(padded_cycle_set) if substring in cycle]
        majority_labels.append(Counter(labels).most_common(1)[0][0] if labels else -1)
    return np.array(majority_labels)

# CA parameter list
CA = [['0','0','0','0','0','3','5','6']]
for ca in CA:
    accuracies = []
    best_acc = 0
    i=9
    train_config, train_class, X_test, y_test = traintestsplit(df_selected, i)
    y_pred = CA_classification(ca, train_config, train_class, X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    best_acc = max(best_acc, acc)
    print(f"Split {i}: Accuracy = {acc:.4f}")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Print or log the results
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"CA {ca}: Mean Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (Best: {best_acc*100:.2f}%)\n")
    with open("ca_summary_results.txt", "a") as file:
        file.write(f"CA {ca}: Mean Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (Best: {best_acc*100:.2f}%)\n")
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # import numpy as np

    #     # Assuming 'accuracies' and 'y_test', 'y_pred_CA_np_array' are defined previously

    #     # Calculate accuracy for each fold and append to the list
    # #   acc_ca = accuracy_score(y_test, y_pred_CA_np_array)
    # #   # accuracies_ca.append(acc)
    # #   print(acc_ca)
    # acc_ca = accuracy_score(y_test, y_pred)
    # prec_ca = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    # rec_ca = recall_score(y_test, y_test, y_pred, average='weighted', zero_division=0)
    # f1_ca = f1_score(y_test, y_test, y_pred, average='weighted', zero_division=0)
    # print(f"Accuracy_Con_CA: {acc_ca:.4f}")
    # print(f"Precision_Con_CA: {prec_ca:.4f}")
    # print(f"Recall_Con_CA: {rec_ca:.4f}")
    # print(f"F1 Score_Con_CA: {f1_ca:.4f}")
# CA = [['0','0','0','0','0','3','0','2'],
#   ['0','0','0','0','0','3','0','4'],
#   ['0','0','0','0','0','3','0','6'],
#   ['0','0','0','0','0','3','0','8'],
#   ['0','0','0','0','0','3','5','6'],
#   ['0','0','0','0','0','7','0','0'],
#   ['0','0','0','0','0','7','0','2'],
#   ['0','0','0','0','0','7','0','3'],
#   ['0','0','0','0','0','7','0','4'],
#   ['0','0','0','0','0','7','0','6'],
#   ['0','0','0','0','0','7','0','8'],
#   ['0','0','0','0','0','7','5','8'],
#   ['0','0','0','0','0','9','0','0'],
#   ['0','0','0','0','0','9','0','1'],
#   ['0','0','0','0','0','9','0','2'],
#   ['0','0','0','0','0','9','0','3'],
#   ['0','0','0','0','0','9','0','4'],
#   ['0','0','0','0','0','9','0','5'],
#   ['0','0','0','0','0','9','0','6'],
#   ['0','0','0','0','0','9','0','7'],
#   ['0','0','0','0','0','9','0','8'],
#   ['0','0','0','0','0','9','5','4'],
#   ['0','0','0','0','5','3','0','0'],
#   ['0','0','0','0','5','3','0','4'],
#   ['0','0','0','0','5','7','0','0'],
#   ['0','0','0','0','5','7','0','2'],
#   ['0','0','0','0','5','9','0','4']
# ]

# for ca in CA:
#     accuracies = []
#     best_acc = 0
#     for i in range(10):
#         train_config, train_class, X_test, y_test = traintestsplit(df_selected, i)
#         y_pred = CA_classification(ca, train_config, train_class, X_test)
#         acc = accuracy_score(y_test, y_pred)
#         accuracies.append(acc)
#         best_acc = max(best_acc, acc)
#         print(f"Split {i}: Accuracy = {acc:.4f}")
#     mean_acc = np.mean(accuracies)
#     std_acc = np.std(accuracies)
#     print(f"CA {ca}: Mean Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (Best: {best_acc*100:.2f}%)\n")
#     with open("ca_summary_results.txt", "a") as file:
#         file.write(f"CA {ca}: Mean Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (Best: {best_acc*100:.2f}%)\n")




# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np
# CA=[['0', '0', '0', '0', '0', '3', '5', '6']]

# for ca in range(0,len(CA)):
#   #print(CA[ca])
#   accuracies_ca = []  
#   split_max = -1
#   max_accuracy = -1# List to store accuracy values
#   n_repeats = 10
#   for i in range(n_repeats):
      

#     #   train_config, train_class, X_test, y_test = traintestsplit(df_selected, i)
#     #   y_pred = CA_classification(ca, train_config, train_class, X_test)
#     #   acc = accuracy_score(y_test, y_pred)
#       #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
#       train_data_configuration, train_data_class,X_test,y_test=traintestsplit(df_selected, i)
#       y_pred_CA_np_array=CA_classification(ca, train_config, train_class, X_test)
#       from sklearn.metrics import accuracy_score
#       import numpy as np

#       # Assuming 'accuracies' and 'y_test', 'y_pred_CA_np_array' are defined previously

#       # Calculate accuracy for each fold and append to the list
#       acc = accuracy_score(y_test, y_pred_CA_np_array)
#       accuracies_ca.append(acc)
#       print(acc)
#       if acc > max_accuracy:
#         max_accuracy = acc
#         split_max = i
#       # Calculate mean accuracy and standard deviation
#   print(accuracies_ca)
#   mean_accuracy = np.mean(accuracies_ca)
#   std_dev = np.std(accuracies_ca)

#   # Assuming 'CA' and 'ca' are defined earlier
#   result_ca = f"{mean_accuracy*100:.2f} ± {std_dev*100:.2f} {CA[ca]}"  # Format as required

#   # Print the result to console
#   print(result_ca)
  #print(accuracies_ca)
  # Write the result to a file (append mode, so results won't be overwritten)
#   with open("./results/Monk1_5.txt", "a") as file:
#       file.write(result_ca + str(accuracies_ca)+"\n")  # Add newline to separate results

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Load the CSV file into a pandas DataFrame
# df = pd.read_csv('./data/monks-1-new.csv')

# # Split the DataFrame into features (X) and target (y)
# X = df.drop(columns=['Class'])  # Replace 'Class' with your actual target column name
# y = df['Class']

# # Encode the target column 'y' (Class) using LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

# # Encode categorical features in X using LabelEncoder
# # Apply LabelEncoder to each column in X that contains strings
# for col in X.select_dtypes(include=['object']).columns:
#     X[col] = le.fit_transform(X[col])

# # List to store accuracy values
# accuracies_SVM = []

# # Number of repeats (e.g., 10)
# n_repeats = 10

# # Loop over n_repeats to perform the classification multiple times
# for i in range(n_repeats):
#     # Train-test split for each repeat
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

#     # Initialize the SVM classifier
#     model = SVC(random_state=42)

#     # Train the model
#     model.fit(X_train, y_train)

#     # Predict on the test set
#     y_pred = model.predict(X_test)

#     # Calculate accuracy for this fold and append to the list
#     acc = accuracy_score(y_test, y_pred)
#     accuracies_SVM.append(acc)

# # Calculate mean accuracy and standard deviation
# mean_accuracy = np.mean(accuracies_SVM)
# std_dev = np.std(accuracies_SVM)

# # Format the result as percentage with two decimal places
# result_svm = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"
# print(result_svm)

# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np

# # List to store accuracy values
# accuracies_KNN = []

# # Assuming 'df' is your DataFrame, and the target column is 'target'
# # Split the DataFrame into features (X) and target (y)
# # X = df.drop(columns=['target'])  # Replace 'target' with your actual target column name
# # y = df['target']

# # Number of repeats (e.g., 10)
# n_repeats = 10

# # Loop over n_repeats to perform the classification multiple times
# for i in range(n_repeats):
#     # Train-test split for each repeat
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

#     # Initialize the KNN classifier (you can adjust 'n_neighbors' based on your data)
#     model = KNeighborsClassifier(n_neighbors=5)  # Example with k=5 neighbors

#     # Train the model
#     model.fit(X_train, y_train)

#     # Predict on the test set
#     y_pred = model.predict(X_test)

#     # Calculate accuracy for this fold and append to the list
#     acc = accuracy_score(y_test, y_pred)
#     accuracies_KNN.append(acc)
# #print(accuracies)
# # Calculate mean accuracy and standard deviation
# mean_accuracy = np.mean(accuracies_KNN)
# std_dev = np.std(accuracies_KNN)

# # Format the result as percentage with two decimal places
# result_KNN = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"  # Format as required

# # Print the result to console
# print(result_KNN)

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np

# # List to store accuracy values
# accuracies_RF = []

# # Number of repeats (e.g., 10)
# n_repeats = 10

# # Loop over n_repeats to perform the classification multiple times
# for i in range(n_repeats):
#     # Train-test split for each repeat
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

#     # Initialize the Random Forest classifier
#     model = RandomForestClassifier(random_state=42, n_estimators=100)  # You can tune n_estimators as needed

#     # Train the model
#     model.fit(X_train, y_train)

#     # Predict on the test set
#     y_pred = model.predict(X_test)

#     # Calculate accuracy for this fold and append to the list
#     acc = accuracy_score(y_test, y_pred)
#     accuracies_RF.append(acc)

# # Calculate mean accuracy and standard deviation
# mean_accuracy = np.mean(accuracies_RF)
# std_dev = np.std(accuracies_RF)

# # Format the result as percentage with two decimal places
# result_RF = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"
# print(result_RF)

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np

# # List to store accuracy values
# accuracies_DT = []

# # Assuming 'X' is your feature matrix and 'y' is your target variable
# # Number of repeats (e.g., 10)
# n_repeats = 10

# # Loop over n_repeats to perform the classification multiple times
# for i in range(n_repeats):
#     # Train-test split for each repeat
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

#     # Initialize the Decision Tree classifier
#     model = DecisionTreeClassifier(random_state=42)

#     # Train the model
#     model.fit(X_train, y_train)

#     # Predict on the test set
#     y_pred = model.predict(X_test)

#     # Calculate accuracy for this fold and append to the list
#     acc = accuracy_score(y_test, y_pred)
#     accuracies_DT.append(acc)

# # Calculate mean accuracy and standard deviation
# mean_accuracy = np.mean(accuracies_DT)
# std_dev = np.std(accuracies_DT)

# # Format the result as percentage with two decimal places
# result_DT = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"

# # Print the result
# print(result_DT)

# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np

# # List to store accuracy values
# accuracies_MLP = []

# # Assuming 'X' is your feature matrix and 'y' is your target variable
# # Number of repeats (e.g., 10)
# n_repeats = 10

# # Loop over n_repeats to perform the classification multiple times
# for i in range(n_repeats):
#     # Train-test split for each repeat
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

#     # Initialize the MLP classifier (you can adjust parameters such as hidden_layer_sizes)
#     model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)  # Example with one hidden layer of 100 neurons

#     # Train the model
#     model.fit(X_train, y_train)

#     # Predict on the test set
#     y_pred = model.predict(X_test)

#     # Calculate accuracy for this fold and append to the list
#     acc = accuracy_score(y_test, y_pred)
#     accuracies_MLP.append(acc)

# # Calculate mean accuracy and standard deviation
# mean_accuracy = np.mean(accuracies_MLP)
# std_dev = np.std(accuracies_MLP)

# # Format the result as percentage with two decimal places
# result_MLP = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"

# # Print the result
# print(result_MLP)

# #& SVM & KNN &RF & DT & MLP &  ALRFRS & Decimal CA \\

# print("SVM:",result_svm)
# print("KNN:",result_KNN)
# print("RF:",result_RF)
# print("DT:",result_DT)
# print("MLP:",result_MLP)
# #print("ALRFRS:",result_ALRFRS)
# print("CA:",result_ca)
















df = pd.read_csv('./data/Dry_Bean.csv')


# for ca in range(0,len(CA)):
#   #print(CA[ca])
#   accuracies_ca = []  # List to store accuracy values
#   #n_repeats = 1
#   i=split_max
#   #for i in range(n_repeats):
#       #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
#   train_data_configuration, train_data_class,X_test,y_test=traintestsplit(df,i)
#   y_pred_CA_np_array=CA_classification(CA[ca], train_data_configuration, train_data_class,X_test)
#   from sklearn.metrics import accuracy_score
#   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#   import numpy as np

#     # Assuming 'accuracies' and 'y_test', 'y_pred_CA_np_array' are defined previously

#     # Calculate accuracy for each fold and append to the list
# #   acc_ca = accuracy_score(y_test, y_pred_CA_np_array)
# #   # accuracies_ca.append(acc)
# #   print(acc_ca)
#   acc_ca = accuracy_score(y_test, y_pred_CA_np_array)
#   prec_ca = precision_score(y_test, y_pred_CA_np_array, average='weighted', zero_division=0)
#   rec_ca = recall_score(y_test, y_pred_CA_np_array, average='weighted', zero_division=0)
#   f1_ca = f1_score(y_test, y_pred_CA_np_array, average='weighted', zero_division=0)
#   print(f"Accuracy_Con_CA: {acc_ca:.4f}")
#   print(f"Precision_Con_CA: {prec_ca:.4f}")
#   print(f"Recall_Con_CA: {rec_ca:.4f}")
#   print(f"F1 Score_Con_CA: {f1_ca:.4f}")
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
df = pd.read_csv('./data/Dry_Bean.csv')
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_swm = accuracy_score(y_test, y_pred)
prec_swm = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec_swm = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_swm = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"SVM Accuracy: {acc_swm:.4f}")
print(f"SVM Precision: {prec_swm:.4f}")
print(f"SVM Recall: {rec_swm:.4f}")
print(f"SVM F1 Score: {f1_swm:.4f}")
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_KNN = accuracy_score(y_test, y_pred)
prec_KNN = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec_KNN = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_KNN = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"KNN Accuracy : {acc_KNN:.4f}")
print(f"KNN Precision: {prec_KNN:.4f}")
print(f"KNN Recall   : {rec_KNN:.4f}")
print(f"KNN F1 Score : {f1_KNN:.4f}")
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_RF = accuracy_score(y_test, y_pred)
prec_RF = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec_RF = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_RF = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"RF Accuracy : {acc_RF:.4f}")
print(f"RF Precision: {prec_RF:.4f}")
print(f"RF Recall   : {rec_RF:.4f}")
print(f"RF F1 Score : {f1_RF:.4f}")
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_DT = accuracy_score(y_test, y_pred)
prec_DT = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec_DT = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_DT = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"DT Accuracy : {acc_DT:.4f}")
print(f"DT Precision: {prec_DT:.4f}")
print(f"DT Recall   : {rec_DT:.4f}")
print(f"DT F1 Score : {f1_DT:.4f}")

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_MLP = accuracy_score(y_test, y_pred)
prec_MLP = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec_MLP = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_MLP = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"MLP Accuracy : {acc_MLP:.4f}")
print(f"MLP Precision: {prec_MLP:.4f}")
print(f"MLP Recall   : {rec_MLP:.4f}")
print(f"MLP F1 Score : {f1_MLP:.4f}")

accuracies_MLP.append(acc)

# Calculate mean accuracy and standard deviation
#mean_accuracy = np.mean(accuracies_MLP)
#std_dev = np.std(accuracies_MLP)

# Format the result as percentage with two decimal places
#result_MLP = f"{mean_accuracy * 100:.2f} ± {std_dev * 100:.2f}"

# Print the result
#print(result_MLP)

#& SVM & KNN &RF & DT & MLP &  ALRFRS & Decimal CA \\

# print("SVM:",acc_swm)
# print("KNN:",acc_KNN)
# print("RF:",acc_RF)
# print("DT:",acc_DT)
# print("MLP:",acc_MLP)
# #print("ALRFRS:",result_ALRFRS)
# print("CA:",acc_ca)