import pandas as pd
from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def convert_to_datetime(timestamp_str):
    for date_format in date_formats:
        try:
            return datetime.strptime(timestamp_str, date_format)
        except ValueError:
            pass
    return None

def calculate_numerical_value(date_str):
    # Split the date string into date and time parts
    date_parts = date_str.split()
    date_part = date_parts[0]
    time_part = date_parts[1]

    # Check if the date part contains hyphens or slashes
    if '-' in date_part:
        year, month, day = map(int, date_part.split('-'))
    elif '/' in date_part:
        year, month, day = map(int, date_part.split('/'))

    # Extract hour, minute, and second components from the time part
    hour, minute, second = map(int, time_part.split(':'))

    # Calculate the numerical representation
    numerical_value = year + (month / 12) + (day / 365) + (hour / 8760) + (minute / 525600) + (second / 31536000)
    
    return numerical_value



df=pd.read_csv('past_data.csv')
df.dropna()


date_formats = ['%d/%m/%Y %H:%M', '%d-%m-%y %H:%M:%S']  
df['IN_TIME'] = df['IN_TIME'].apply(convert_to_datetime)
df['OUT_TIME'] = df['OUT_TIME'].apply(convert_to_datetime)

# Calculate the duration in hours
df['DURATION'] = (df['OUT_TIME'] - df['IN_TIME']).dt.total_seconds() / 3600  # Convert to hours

# Print the updated DataFrame
df.drop(["REF_ID","CON_NUM","VALIDITY"],axis=1,inplace=True)
df = df[df['DURATION'] >= 0]
df.drop(["DURATION"],axis=1,inplace=True)
df=pd.get_dummies(data=df,drop_first=True)


df['IN_TIME_NUMERIC'] = df['IN_TIME'].astype(str).apply(lambda date_str: calculate_numerical_value(date_str))


df['OUT_TIME_NUMERIC'] = df['OUT_TIME'].astype(str).apply(lambda date_str: calculate_numerical_value(date_str))
df.dropna(inplace=True)
x=df[["IN_TIME_NUMERIC","CON_SIZE","STATUS_L"]]
y=df["OUT_TIME_NUMERIC"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

print('MAE: ',metrics.mean_absolute_error(y_test,y_pred))
print('MSE: ',metrics.mean_squared_error(y_test,y_pred))
print('R2', metrics.r2_score(y_test,y_pred))

ind=pd.read_csv('indata.csv')
ind.dropna()
ind.drop(["REF_ID","ID","CON_NUM"],axis=1,inplace=True)
ind['IN_TIME'] = ind['IN_TIME'].apply(convert_to_datetime)
ind['IN_TIME_NUMERIC'] = ind['IN_TIME'].astype(str).apply(lambda date_str: calculate_numerical_value(date_str))
ind.drop(["IN_TIME"],axis=1,inplace=True)
ind=pd.get_dummies(data=ind,drop_first=True)
X=ind[["IN_TIME_NUMERIC","CON_SIZE","STATUS_L"]]
ind['OUT_TIME_PREDICTED'] = lr.predict(X)
print(ind)
# Sort the containers by predicted leave time
ind.sort_values(by='OUT_TIME_PREDICTED', ascending=True, inplace=True)

# Initialize variables to track the current block, row, bay, and level
current_block = 1
current_row = 'A'
current_bay = 1
current_level = 1

# Function to calculate the next location
def assign_locations(df):
    # Initialize variables to track block, row, bay, and level
    current_block = 1
    current_row = 'A'
    current_bay = 1
    current_level = 1

    locations = []  # List to store assigned locations

    for index, row in df.iterrows():
        # Check if the block is full (3 levels)
        if current_level > 3:
            current_level = 1  # Reset level
            current_bay += 1  # Move to the next bay

            # Check if the bay is full (10 bays)
            if current_bay > 9:
                current_bay = 1  # Reset bay
                current_row = chr(ord(current_row) + 1)  # Move to the next row

                # Check if the row is full (A to C)
                if current_row > 'C':
                    current_row = 'A'  # Reset row
                    current_block += 1  # Move to the next block

        # Generate the location based on block, row, bay, and level
        location = f'B{current_block}{current_row}{current_bay}{current_level}'
        locations.append(location)

        # Update the current level
        current_level += 1

    df['LOCATION'] = locations

# Call the function to assign locations to containers in 'ind'
assign_locations(ind)

# Print the DataFrame with assigned locations
new_index = range(1, len(ind) + 1)

# Assign the new index to the DataFrame
ind.index = new_index
ind.index.name = 'ID'
print(ind[['CON_SIZE', 'IN_TIME_NUMERIC', 'STATUS_L', 'OUT_TIME_PREDICTED', 'LOCATION']])
ind.to_csv("with pos.csv")


