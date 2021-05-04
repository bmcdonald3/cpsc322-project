import mysklearn.myutils as myutils
import copy
import csv 
#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            tuple of int: rows, cols in the table

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.column_names.index(col_identifier)
        column = []
        
        if not include_missing_values:
            for row in self.data:
                if row[col_index] != 'NA':
                    column.append(row[col_index])
        else:
            for row in self.data:
                column.append(row[col_index])

        return column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        data_copy = []
        indices_to_skip = []
        for row in rows_to_drop:
            try:
                indices_to_skip.append(self.data.index(row))
            except:
                pass
        for i in range(len(self.data)):
            if i not in indices_to_skip:
                data_copy.append(self.data[i])
        
        self.data = data_copy

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, 'r')
        filereader = csv.reader(infile, dialect='excel')
        table = []

        for row in filereader:
            if len(row) > 0:
                table.append(row)

        header = table[0]
        del table[0]

        infile.close()

        self.column_names = header
        self.data = table
        self.convert_to_numeric()

        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, 'w')
        for i in range(len(self.column_names)-1):
            outfile.write(str(self.column_names[i]) + ',')
        outfile.write(str(self.column_names[-1]) + '\n')
        for row in self.data:
            for i in range(len(row)-1):
                outfile.write(str(row[i]) + ',')
            outfile.write(str(row[i+1]) + '\n')
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        dups = []
        cols = []
        for val in key_column_names:
            cols.append(self.column_names.index(val))

        cols_copy = []
        for i in range(len(self.data)):
            current = []
            for j in cols:
                current.append(self.data[i][j])
            cols_copy.append(current)

        for i in range(len(cols_copy)-1):
            if cols_copy[i] in cols_copy[i+1:]:
                dups.append(self.data[cols_copy[i+1:].index(cols_copy[i])+i+1])

        return dups

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        table_copy = []
        for row in self.data:
            if 'NA' not in row:
                table_copy.append(row)
        self.data = table_copy

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        self.convert_to_numeric
        col = self.get_column(col_name, False)
        avg = sum(col)/len(col)

        col = self.get_column(col_name)

        for i in range(len(self.data)):
            if col[i] == 'NA':
                self.data[i][col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """
        table = []
        for val in col_names:
            try:
                col = self.get_column(val,False)
                row = [val]
                row.append(min(col))
                row.append(max(col))
                row.append((min(col)+max(col))/2)
                row.append(sum(col)/len(col))
                col.sort()
                if len(col) % 2 == 0:
                    m1 = col[len(col)//2]
                    m2 = col[len(col)//2 - 1]
                    row.append((m1+m2)/2)
                else:
                    row.append(col[len(col)//2])

                table.append(row)
            except ValueError:
                pass

        return MyPyTable(['attribute','min','max','mid','avg','median'], table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        new_header = []
        for val in self.column_names:
            new_header.append(val)
        for val in other_table.column_names:
            if val not in new_header:
                new_header.append(val)
        
        cols1 = []
        cols2 = []
        for val in key_column_names:
            cols1.append(self.column_names.index(val))
            cols2.append(other_table.column_names.index(val))
        
        keys1 = []
        keys2 = []
        for i in range(len(self.data)):
            current = []
            for j in cols1:
                current.append(self.data[i][j])
            keys1.append(current)
        for i in range(len(other_table.data)):
            current = []
            for j in cols2:
                current.append(other_table.data[i][j])
            keys2.append(current)
        
        joined_table = []
        for i in range(len(keys1)):
            if keys1[i] in keys2: # keys match so make new row of the combined stuff
                current = copy.deepcopy(self.data[i])
                for val in other_table.column_names:
                    if val not in self.column_names: # values not in the first table
                        current.append(other_table.data[keys2.index(keys1[i])][other_table.column_names.index(val)]) # doesnt work since i corresponds to first table
                joined_table.append(current)

        return MyPyTable(new_header, joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_header = []
        for val in self.column_names:
            new_header.append(val)
        for val in other_table.column_names:
            if val not in new_header:
                new_header.append(val)
        
        cols1 = []
        cols2 = []
        for val in key_column_names:
            cols1.append(self.column_names.index(val))
            cols2.append(other_table.column_names.index(val))
        
        keys1 = []
        keys2 = []
        for i in range(len(self.data)):
            current = []
            for j in cols1:
                current.append(self.data[i][j])
            keys1.append(current)
        for i in range(len(other_table.data)):
            current = []
            for j in cols2:
                current.append(other_table.data[i][j])
            keys2.append(current)
        
        num_extra_rows = len(new_header) - len(self.column_names)

        joined_table = []
        for i in range(len(keys1)):
            current = copy.deepcopy(self.data[i])
            if keys1[i] in keys2: # keys match so make new row of the combined stuff
                for val in other_table.column_names:
                    if val not in self.column_names: # values not in the first table
                        current.append(other_table.data[keys2.index(keys1[i])][other_table.column_names.index(val)]) # doesnt work since i corresponds to first table  
            else: # keys don't match so add in row with 'NA's
                for i in range(num_extra_rows):
                    current.append('NA')
            joined_table.append(current)

        # add rows in second table that didn't make it to the first one
        for i in range(len(keys2)):
            if keys2[i] not in keys1: # keys don't match, so haven't added row yet
                current = []
                for col_name in new_header:
                    if col_name in other_table.column_names: # is in the table we are checking so put in that value
                        current.append(other_table.data[i][other_table.column_names.index(col_name)])
                    else:
                        current.append('NA') # not in the first table so add 'NA'  
                joined_table.append(current)

        return MyPyTable(new_header, joined_table)
    