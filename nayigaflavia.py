#import  the require libraries#import  the require libraries
import pandas as pd  
from mlxtend.preprocessing import TransactionEncoder  
from mlxtend.frequent_patterns import apriori, association_rules 
#PART A:DATA PREPROCESSING
print("PART A: DATA PREPARATION")  
print()  
# Create a list of transactions
transactions = [
    ['Bread', 'Milk', 'Eggs'],          
    ['Bread', 'Butter'],               
    ['Milk', 'Diapers', 'Beer'],        
    ['Bread', 'Milk', 'Butter'],       
    ['Milk', 'Diapers', 'Bread'],      
    ['Beer', 'Diapers'],                
    ['Bread', 'Milk', 'Eggs', 'Butter'],
    ['Eggs', 'Milk'],                  
    ['Bread', 'Diapers', 'Beer'],      
    ['Milk', 'Butter']                  
]
print("Transaction Dataset") 
# Create a pandas DataFrame with transaction IDs and item lists
df = pd.DataFrame({  
    'Transaction_ID': range(1, 11), 
    'Items': [', '.join(items) for items in transactions] 
})
# Print the DataFrame without showing index numbers
print(df.to_string(index=False)) 
print() 
# creating one hot encoded format
print("One-Hot Encoded Format") 
encoder = TransactionEncoder() 
encoder_arry = encoder.fit(transactions).transform(transactions)
df = pd.DataFrame(encoder_arry, columns=encoder.columns_) 
print(df) 
print()
# PART B: APRIORI ALGORITHM 
print("PART B: APRIORI ALGORITHM") 
print("Minimum Support: 0.2")
print(" Minimum Confidence: 0.5")
print()
# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print("Frequent Itemsets Found") 
print(frequent_itemsets) 
print()
# Generate association rules from frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("Association Rules Generated:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print() 


