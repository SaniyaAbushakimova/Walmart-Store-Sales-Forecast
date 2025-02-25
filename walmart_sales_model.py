#-------------------------------------------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------------------------------------------
import pandas as pd
import argparse
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
import patsy

#-------------------------------------------------------------------------------------------------------------------
# Helper Function Definitions
#-------------------------------------------------------------------------------------------------------------------
# SVD Helper functions
# Pivot function to Get spread matrix from the data
def pivot_X(data):
  '''
  Input:
    data: Unprocessed orginal Dataframe, Note: don't pass output of preprocess to this function
  Returns:
    X: The processed pivoted dataframe, with NA values filled with 0
  '''
  selected_columns = data[['Store', "Dept", "Date", 'Weekly_Sales']]

  X = selected_columns.pivot(index=['Store', 'Dept'], columns=["Date"], values='Weekly_Sales').reset_index()

  X.fillna(0, inplace=True)

  return X

# Function to run SVD for an individual department 
def dept_SVD(X, dept, d):
  '''
  Input:
    X: The Unprocessed Pivoted Dataframe (output of Pivot X)
    dept: The department for which to get the SVD Reduced Matrix 
    d: The number of top diagonal values to create the reduction

  Returns:
    X_red: The DataFrame in the original format with SVD reduced values, for the specifc department
  '''

  # Filter the department
  X_dept = X[X['Dept'] == dept].reset_index(drop=True)

  # Keep Cols for reconstructing matrix later, keep only weekly sale values for SVD
  cols = X_dept.columns
  store_dept_vals = X_dept[['Store', 'Dept']].values
  X_dept = X_dept.drop(columns=['Store', 'Dept'])

  # Center the data
  means = np.mean(X_dept.values , axis=1, keepdims=True)
  X_dept = X_dept - means

  #  SVD keep top d diagonal values
  U, D, Vt = np.linalg.svd(X_dept, full_matrices=False)

  D_red = np.zeros_like(D)
  D_red[:d] = D[:d]

  # Reconstructing Reduced X & recentering
  X_red = U @ np.diag(D_red) @ Vt
  X_red = X_red + means

  # Add back the cols and convert back to data frame
  X_red = np.hstack((store_dept_vals, X_red))

  df = pd.DataFrame(X_red, columns=cols).reset_index(drop=True)
  df[["Store", "Dept"]] = df[["Store", "Dept"]].astype(int)

  return df.melt(id_vars=["Store", "Dept"], var_name="Date", value_name="Weekly_Sales")


# Pre-Processing Function to Change Date in Wk & Yr Colmuns
def preprocess(data):

    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks

    return data


# Shift/Post Adjustment Function
def post_adjustment(test_pred, threshold=1.1, shift_days=1):

    test_pred["Date"] = pd.to_datetime(test_pred["Date"])
    test_pred["Wk"] = test_pred["Date"].dt.isocalendar().week

    shift_fraction = shift_days / 7

    for dept in test_pred["Dept"].unique():
        dept_data = test_pred[(test_pred["Dept"] == dept) & (test_pred["Wk"].isin([48, 49, 50, 51, 52]))].copy()

        # Average sales for weeks 48 and 52
        baseline = dept_data[dept_data["Wk"].isin([48, 52])]["Weekly_Pred"].mean()

        # Average sales for weeks 49, 50, 51
        surge = dept_data[dept_data["Wk"].isin([49, 50, 51])]["Weekly_Pred"].mean()

        if baseline > 0 and (surge / baseline) > threshold:
            weeks = dept_data["Wk"].values
            shifted_sales = dept_data["Weekly_Pred"].values.copy()

            for i in range(1, len(shifted_sales)):
                shifted_sales[i] = (1 - shift_fraction) * shifted_sales[i] + shift_fraction * shifted_sales[i - 1]
            shifted_sales[0] = dept_data["Weekly_Pred"].values[0]  # No change to the first week in the sequence (week 48)

            test_pred.loc[dept_data.index, "Weekly_Pred"] = shifted_sales

    return test_pred

#-------------------------------------------------------------------------------------------------------------------
# Main Training Loop 
#-------------------------------------------------------------------------------------------------------------------

# Argument parser
parser = argparse.ArgumentParser(description="Train a model using given train and test datasets.")
parser.add_argument("--train", type=str, required=True, help="Path to the training dataset (CSV)")
parser.add_argument("--test", type=str, required=True, help="Path to the test dataset (CSV)")

# Parse arguments
args = parser.parse_args()

# Load and read csv files
train = pd.read_csv(args.train)
test = pd.read_csv(args.test)

# Remove noise with SVD
test_depts = test['Dept'].unique()
train_new = pd.DataFrame()
X = pivot_X(train) # Keep outside of loop to save on computation time

for dept in test_depts:
  train_reduced = dept_SVD(X, dept, 8)
  train_new = pd.concat([train_new, train_reduced], ignore_index=True)

train = train.merge(train_new[['Store', 'Dept', 'Date', 'Weekly_Sales']],
                      on=['Store', 'Dept', 'Date'], how='left', suffixes=('', '_new'))
train = (train.drop(columns=['Weekly_Sales']).
          rename(columns={'Weekly_Sales_new': 'Weekly_Sales'}))

# pre-allocate a pd to store the predictions
test_pred = pd.DataFrame()

train_pairs = train[['Store', 'Dept']].drop_duplicates(ignore_index=True)
test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
unique_pairs = pd.merge(train_pairs, test_pairs, how = 'inner', on =['Store', 'Dept'])

train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')
train_split = preprocess(train_split)
X = patsy.dmatrix('Weekly_Sales + Store + Dept + Yr  + Wk + I(Yr**2)',
                    data = train_split,
                    return_type='dataframe')
train_split = dict(tuple(X.groupby(['Store', 'Dept'])))


test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
test_split = preprocess(test_split)
X = patsy.dmatrix('Store + Dept + Yr  + Wk + I(Yr**2)',
                      data = test_split,
                      return_type='dataframe')
X['Date'] = test_split['Date']
test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

keys = list(train_split)

for key in keys:
  X_train = train_split[key]
  X_test = test_split[key]

  Y = X_train['Weekly_Sales']
  X_train = X_train.drop(['Weekly_Sales','Store', 'Dept'], axis=1)
  cols_to_drop = X_train.columns[(X_train == 0).all()]
  X_train = X_train.drop(columns=cols_to_drop)
  X_test = X_test.drop(columns=cols_to_drop)

  cols_to_drop = []
  for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
      col_name = X_train.columns[i]
      # Extract the current column and all previous columns
      tmp_Y = X_train.iloc[:, i].values
      tmp_X = X_train.iloc[:, :i].values
  
      coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
      if np.sum(residuals) < 1e-16:
            cols_to_drop.append(col_name)
  
  X_train = X_train.drop(columns=cols_to_drop)
  X_test = X_test.drop(columns=cols_to_drop)

  model = sm.OLS(Y, X_train).fit()
  mycoef = model.params.fillna(0)

  tmp_pred = X_test[['Store', 'Dept', 'Date']]
  X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)

  tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
  test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

test_pred = test.merge(test_pred, on=['Dept', 'Store', 'Date'], how='left')
test_pred['Weekly_Pred'] = test_pred["Weekly_Pred"].fillna(0)

# Check if we are in between fold 5 dates in order to do Post-process
if pd.to_datetime(test["Date"])[0] > pd.to_datetime("2011-11-01") and pd.to_datetime(test['Date'])[len(test["Date"]) - 1] < pd.to_datetime("2012-01-01"):
    test_pred = post_adjustment(test_pred)

test_pred.to_csv('mypred.csv', index=False)