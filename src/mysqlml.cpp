/*
Copyright (C) 2013 Pranjal Mittal, Abinash Panda

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


/********************************************************************
Pranjal Mittal, Abinash Panda
Using Principal Component Analysis for feature reduction (Machine Learning Algorithm) 
to process MySQL Data on a server.

Requirements:


Compile: 
g++ mysqlml.cpp -lmysqlcppconn -lgsl -lgslcblas

Using Program: ./a.out DB_HOST DB_USER DB_PASS DB_NAME
********************************************************************/

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
	
//MySQL Connector Specific Headers & Settings
#include "mysql_connection.h"

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

// [IMP] Please set appropriately as per your database settings.
#define DB_HOST "localhost"
#define DB_USER "root"
#define DB_PASS "iinetctp"
#define DB_NAME "mydb_ml"

//GNU Scientific Library (GSL) specific.
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>

//Self Written Header for Principal Component Analysis.
#include "pca.h"

//Number of Columns in Non-Reduced Dataset
#define COLUMNS 100
#define ROWS 100

using namespace std;

int main(int argc, const char **argv)
{	

    string url(argc >= 2 ? argv[1] : DB_HOST);
    const string user(argc >= 3 ? argv[2] : DB_USER);
    const string pass(argc >= 4 ? argv[3] : DB_PASS);
    const string database(argc >= 5 ? argv[4] : DB_NAME);

    cout << "********************************************" << endl;
    cout << "Processing MySQL data using Machine Learning" << endl;
    cout << "********************************************" << endl;
    cout << "|       Principal Component Analysis       |"  << endl;
    cout << "--------------------------------------------" << endl;
    cout << "Reducing dimension of input feature set..." << endl;

	// Try to connect and run and run any code.
    try {

        sql::Driver * driver = get_driver_instance();

        std::auto_ptr< sql::Connection > con(driver->connect(url, user, pass));
        con->setSchema(database);

        std::auto_ptr< sql::PreparedStatement >  pstmt;

        sql::Statement *stmt;
        sql::ResultSet  *res;
        stmt = con->createStatement();

        gsl_matrix * m = gsl_matrix_alloc (ROWS,COLUMNS);

        res = stmt->executeQuery("SELECT * FROM pca;");

        // Populating the GSL Data Matrix to be used in the PCA Algorithm.
        int i = 0;
        while (res->next()) {
            if (i >= ROWS){
                break;
            }
            for (int j=0;j<COLUMNS;j++){
                // cout << res->getDouble(j+1) << endl; 
                // getDouble(1) returns the first column (converting FLOAT in MySQL to Double in Cpp)
                gsl_matrix_set(m, i, j, res->getDouble(j+1));

                // res->getString("<column_name>")
            }
            i++;
            // cout << endl; 
        }

        // GSL Data matrix can be used for ML analysis here:
    	int k;
        gsl_matrix *red_data;
        red_data = pca(m,ROWS,COLUMNS,&k);

        cout << "Reduced Dimension:" << k << endl;
         
        //// Test Code to print the Matrix read from the MySQL file.
        for (int i = 0; i < ROWS; i++){
            for (int j = 0; j < k; j++)
                printf ("%g ",gsl_matrix_get (m, i, j));
            cout << endl << endl;
        }        

        delete res;
        delete stmt;
	
    } catch (sql::SQLException &e) {
        /*
          The MySQL Connector/C++ throws three different exceptions:
	
          - sql::MethodNotImplementedException (derived from sql::SQLException)
          - sql::InvalidArgumentException (derived from sql::SQLException)
          - sql::SQLException (derived from std::runtime_error)
        */
        cout << "# ERROR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        /* what() (derived from std::runtime_error) to fetch the error message */
        cout << "# ERROR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
	
        return EXIT_FAILURE;
    }
	
    cout << "Program Successfuly Completed." << endl;
    return EXIT_SUCCESS;
}
