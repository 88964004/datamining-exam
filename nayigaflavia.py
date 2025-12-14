import pandas as pd  
from mlxtend.preprocessing import TransactionEncoder  
from mlxtend.frequent_patterns import apriori, association_rules 

print("PART A: DATA PREPARATION")  
print()  
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
df = pd.DataFrame({  
    'Transaction_ID': range(1, 11), 
    'Items': [', '.join(items) for items in transactions] 
})
print(df.to_string(index=False)) 
print() 

print("One-Hot Encoded Format")  
te = TransactionEncoder() 
te_arry = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_arry, columns=te.columns_) 
print(df) 

print()  
print("APRIORI ALGORITHM") 
print("Minimum Support: 0.2")
print(" Minimum Confidence: 0.5")
print()
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print("Frequent Itemsets Found") 
print(frequent_itemsets) 
print()

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("Association Rules Generated:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print() 
