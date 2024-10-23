import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from the correct path
data = pd.read_csv("C:/Users/ASUS/OneDrive/Documents/DATAMINING_all_answer/Data_Mining_Slips-main/Slip_no_15/shows.csv")  # Ensure this file is in the correct directory

# Preprocess the data: Map 'Nationality' to numeric values
data['Nationality'] = data['Nationality'].apply(lambda x: 1 if x == 'American' else 0)

# Define features and target variable
X = data[['Age', 'Nationality', 'Experience', 'Rank']]
y = data['Go']

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Fit the classifier to the data
clf = clf.fit(X, y)

# Create an input data for prediction
input_data = pd.DataFrame({'Age': [40], 'Nationality': [1], 'Experience': [10], 'Rank': [7]})

# Predict the class label
predicted_label = clf.predict(input_data)

# Create a DataFrame to store the input and predicted label
result_df = pd.DataFrame({
    'Age': input_data['Age'],
    'Nationality': ['American' if input_data['Nationality'].values[0] == 1 else 'Other'],
    'Experience': input_data['Experience'],
    'Rank': input_data['Rank'],
    'Predicted Label': predicted_label
})

# Save the result to a CSV file
result_df.to_csv(r'Slip_no_15\predicted_show_label.csv', index=False)
print("Predicted Label:", predicted_label)
print("File path to shows.csv:")
print(r"C:\Users\ASUS\OneDrive\Documents\DATAMINING_all_answer\Data_Mining_Slips-main\Slip_no_15\shows.csv")
