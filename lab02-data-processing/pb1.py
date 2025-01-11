import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# 1. Print the number of employees
def nr_of_employees(df_employees):
    return df_employees.count().max()


# 2. Print columns and their data types
def columns_and_types(df_employees):
    df_cols_types = []
    for col in df_employees.columns:
        df_cols_types.append([col, df_employees[col].dtype])
    return df_cols_types


def nr_proprieties_for_employee(df_employees):
    return len(df_employees.columns)


# 3. Print number of employees with complete data
def nr_employees_with_complete_data(df_employees):
    return df_employees.dropna().count().max()


# 4. Print minimum, maximum, and mean values for each property
def min_max_med_propriety(df_employees):
    return df_employees.describe()


# 5. Print number of possible values for non-numeric properties
def nr_possible_non_numeric_proprieties(df_employees):
    return df_employees.select_dtypes(exclude='number').nunique()


# 6. Print number of null values for each property
def nr_null_values(df_employees):
    return df_employees.isnull().sum()


def pb1a():
    # Read the CSV file
    df_employees = pd.read_csv('data/employees.csv', delimiter=',', header='infer')

    print('1')
    print('Numarul de angajati este: ', nr_of_employees(df_employees))
    print()

    print('2')
    for col_type in columns_and_types(df_employees):
        print('Coloana: ', col_type[0], '\nTipul: ', col_type[1])
    print()
    print('Numarul proprietatilor detinute pentru un angajat: ', nr_proprieties_for_employee(df_employees))
    print()

    print('3')
    print('Numarul de angajati pentru care se detin date complete: ', nr_employees_with_complete_data(df_employees))
    print()

    print('4')
    print('Valorile minime, maxime, medii pentru fiecare proprietate: ')
    print(min_max_med_propriety(df_employees))
    print()

    print('5')
    print('Numar de valori posibile pentru fiecare proprietate nenumerica:\n',
          nr_possible_non_numeric_proprieties(df_employees))
    print()

    print('6')
    print('Numarul de valori null pentru fiecare proprietate: \n', nr_null_values(df_employees))
    print()

    # print('Valorile null sunt: ')
    # print(df_employees[df_employees.isnull().any(axis=1)])

    # df_employees = df_employees.fillna(df_employees.mean())
    # df_employees = df_employees.dropna(axis=0, how='any')

    # print()


pb1a()


def test_pb1a():
    df_employees = pd.read_csv('data/employees.csv', delimiter=',', header='infer')

    assert (nr_of_employees(df_employees) == 1000)
    assert (columns_and_types(df_employees)[0] == ['First Name', object])
    assert (nr_proprieties_for_employee(df_employees) == 8)
    assert (nr_employees_with_complete_data(df_employees) == 764)
    # assert (min_max_med_propriety(df_employees))
    assert (nr_possible_non_numeric_proprieties(df_employees)['Gender'] == 2)
    assert (nr_null_values(df_employees)['Senior Management'] == 67)


# test_pb1a()


def pb1b():
    """
    1. Reads employee data from a CSV file named 'employees.csv' in the 'data' directory.
    2. Filters out rows with missing salary data.
    3. Visualizes the distribution of salaries using a histogram.
    4. Divides salaries into categories: '< 50k', '50k - 100k', '100k - 150k', '150k - 200k', '> 200k'.
    5. Plots the distribution of salaries per category and team using histograms.
    6. Identifies outliers using a cutoff of two standard deviations from the mean salary.
    7. Prints the first names of outliers.

    Dependencies:
    - pandas: for data manipulation.
    - matplotlib.pyplot: for data visualization.

    Returns:
    None
    """
    df_employees = pd.read_csv('data/employees.csv', delimiter=',', header='infer')
    df_employees = df_employees.dropna(subset=['Salary'])

    plt.figure(figsize=(10, 4))
    plt.hist(df_employees['Salary'], bins=10, color='skyblue', edgecolor='black')
    plt.title('distributia salariilor pe categorii de salar')
    plt.xlabel('salariu')
    plt.ylabel('număr de angajați')
    plt.grid(True)
    plt.show()

    salary_bins = [0, 50000, 100000, 150000, 200000, float('inf')]
    salary_labels = ['< 50k', '50k - 100k', '100k - 150k', '150k - 200k', '> 200k']
    df_employees['Salary Category'] = pd.cut(df_employees['Salary'], bins=salary_bins, labels=salary_labels)

    plt.figure(figsize=(10, 6))
    for team, data in df_employees.groupby('Team'):
        plt.hist(data['Salary'], bins=20, alpha=0.5, label=str(team), density=True)

    plt.title('distribuția salariilor pe categorii de salar și echipă')
    plt.xlabel('salariu')
    plt.ylabel('densitate')
    plt.legend()
    plt.show()

    mean_salary = df_employees['Salary'].mean()
    std_salary = df_employees['Salary'].std()
    lower_cutoff = mean_salary - 2 * std_salary
    upper_cutoff = mean_salary + 2 * std_salary

    outliers = df_employees[(df_employees['Salary'] < lower_cutoff) | (df_employees['Salary'] > upper_cutoff)]
    print("Outliers: ", outliers['First Name'])


pb1b()
