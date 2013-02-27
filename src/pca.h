/*
Copyright (C) 2013 Pranjal Mittal, Abinash Panda

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#include<iostream>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_linalg.h>
using namespace std;

// Returns the index k at which sum from 0 to k exeeds val.
int sum_till_lessthan(gsl_vector *S,int length, double val){
    int k  = 0;
    double sum = 0;
	for (int i = 0; i<length; i++){
		sum += gsl_vector_get(S,i);
        if (sum >= val){    
            return k;
        }
        k++;
    }
    return k;
}

//function for normalizing the array (x <- (x-mean))
gsl_matrix *normalize(gsl_matrix *mat, int m, int n) {
	gsl_vector *mean = gsl_vector_alloc(n);
	gsl_vector *row_vector = gsl_vector_alloc(n);
	for (int i = 0; i<m; i++) {
		gsl_matrix_get_row(row_vector,mat,i);
		gsl_vector_add(mean,row_vector);
	}
	gsl_vector_scale(mean,1.0/m);
	gsl_matrix *norm_mat = gsl_matrix_alloc(m,n);
	for (int i = 0; i<m; i++) {
		gsl_matrix_get_row(row_vector,mat,i);
		gsl_vector_sub(row_vector,mean);
		gsl_matrix_set_row(norm_mat,i,row_vector);
	}
	return norm_mat;
}

//function for computing the covariance of the array
gsl_matrix *covariance(gsl_matrix *mat, int m, int n) {
	gsl_matrix *row_matrix = gsl_matrix_alloc(1,n);
	gsl_matrix *covar = gsl_matrix_alloc(n,n);
	for (int i = 0; i<m; i++) {
		for (int j = 0; j<n; j++)
			gsl_matrix_set(row_matrix,0,j,gsl_matrix_get(mat,i,j));
		gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,row_matrix,row_matrix,1.0,covar);
	}
	gsl_matrix_scale(covar,1.0/m);
	return covar;
}

//function performing PCA on the given matrix; returns the reduced matrix
gsl_matrix *pca(gsl_matrix *mat, int m, int n,int *k) {
	gsl_matrix *V = gsl_matrix_alloc(n,n);
	gsl_vector *S = gsl_vector_alloc(n);
	gsl_vector *work = gsl_vector_alloc(n);
	gsl_matrix *norm;
	norm = normalize(mat,m,n);

	gsl_matrix *covar;
	covar = covariance(norm,m,n);	
	gsl_linalg_SV_decomp(covar,V,S,work);
	double sum = 0;
	int new_dim = 0;
	for (int i = 0; i<n; i++)
		sum = sum + gsl_vector_get(S,i);

	new_dim = sum_till_lessthan(S,n,(0.99*sum));
	gsl_matrix *dim = gsl_matrix_alloc (n,new_dim);

    //cout << new_dim << endl;

	for(int i = 0; i<n; i++) {
		for (int j = 0; j<new_dim; j++){
			gsl_matrix_set(dim,i,j,gsl_matrix_get(covar,i,j));
        }
	}
    	
    gsl_matrix *red_data = gsl_matrix_alloc(m,new_dim);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,mat,dim,0.0,red_data);
	*k = new_dim;	
	return red_data;
}

/*
int main() {
	double data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
	gsl_matrix_view mat = gsl_matrix_view_array(data,5,10);
	int k = 0;
	gsl_matrix *red_data;
	red_data = pca(&mat.matrix,5,10,&k);
	for (int i = 0; i<5; i++)
		cout<<gsl_matrix_get(red_data,i,0)<<endl; 
	return 0;
}
*/
