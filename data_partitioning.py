import pandas as pd
from utils import seed

# Set a seed for reproducibility
# seed = 42

a = pd.read_csv("./BankA.csv")
b = pd.read_csv("./BankB.csv")
c = pd.read_csv("./BankC.csv")

# Create Test Sets
test_a = a.sample(frac=0.1, random_state=seed)
remaining_a = a.drop(test_a.index)

test_b = b.sample(frac=0.1, random_state=seed)
remaining_b = b.drop(test_b.index)

test_c = c.sample(frac=0.1, random_state=seed)
remaining_c = c.drop(test_c.index)

# Create Validation Sets
validation_a = remaining_a.sample(frac=0.2, random_state=seed)
remaining_a = remaining_a.drop(validation_a.index)

validation_b = remaining_b.sample(frac=0.2, random_state=seed)
remaining_b = remaining_b.drop(validation_b.index)

validation_c = remaining_c.sample(frac=0.2, random_state=seed)
remaining_c = remaining_c.drop(validation_c.index)

# Save the test sets
test_a.to_csv('BankA_Test.csv')
test_b.to_csv('BankB_Test.csv')
test_c.to_csv('BankC_Test.csv')

# Save the validation sets
validation_a.to_csv('BankA_Val.csv')
validation_b.to_csv('BankB_Val.csv')
validation_c.to_csv('BankC_Val.csv')

# Save the remaining data as training sets
remaining_a.to_csv('BankA_Train.csv')
remaining_b.to_csv('BankB_Train.csv')
remaining_c.to_csv('BankC_Train.csv')

# Concatenate the test sets
all_banks_test = pd.concat([test_a, test_b, test_c], ignore_index=True)

# Save the concatenated test set
all_banks_test.to_csv('All_Banks_Test.csv')