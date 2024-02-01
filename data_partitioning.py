import pandas as pd
from utils import seed
from sklearn.model_selection import train_test_split

# Set a seed for reproducibility
# seed = 42

a = pd.read_csv("./BankA.csv")
b = pd.read_csv("./BankB.csv")
c = pd.read_csv("./BankC.csv")

# Set a seed for reproducibility
# seed = 42

a = pd.read_csv("./BankA.csv")
b = pd.read_csv("./BankB.csv")
c = pd.read_csv("./BankC.csv")

# Create Test Sets

# Assuming 'strata_column' is the column you want to stratify on
strata_column = 'income'

# Create Test Sets
train_a, test_a = train_test_split(a, test_size=0.1, random_state=seed, stratify=a[strata_column])
train_b, test_b = train_test_split(b, test_size=0.1, random_state=seed, stratify=b[strata_column])
train_c, test_c = train_test_split(c, test_size=0.1, random_state=seed, stratify=c[strata_column])

# Create Validation Sets
remaining_a, validation_a = train_test_split(train_a, test_size=0.2, random_state=seed, stratify=train_a[strata_column])
remaining_b, validation_b = train_test_split(train_b, test_size=0.2, random_state=seed, stratify=train_b[strata_column])
remaining_c, validation_c = train_test_split(train_c, test_size=0.2, random_state=seed, stratify=train_c[strata_column])

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

# Concatenate the sets
all_banks_train = pd.concat([remaining_a, remaining_b, remaining_c], ignore_index=True)
all_banks_val = pd.concat([validation_a, validation_b, validation_c], ignore_index=True)
all_banks_test = pd.concat([test_a, test_b, test_c], ignore_index=True)

# Save the concatenated set
all_banks_train.to_csv('All_Banks_Train.csv')
all_banks_val.to_csv('All_Banks_Val.csv')
all_banks_test.to_csv('All_Banks_Test.csv')
