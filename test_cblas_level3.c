#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#define EPSF 1e-4
#define EPSD 1e-8



int cmpf(float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        if (fabs(a[i] - b[i]) > EPSF) return 0;
    return 1;
}

int cmpd(double *a, double *b, int n) {
    for (int i = 0; i < n; i++)
        if (fabs(a[i] - b[i]) > EPSD) return 0;
    return 1;
}


int cmpf_tri(float *a, float *b, int N) {
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            if (fabs(a[i*N + j] - b[i*N + j]) > EPSF)
                return 0;
    return 1;
}

void fillf(float *a, int n) {
    for (int i = 0; i < n; i++) a[i] = (float)(i % 7 + 1);
}

void filld(double *a, int n) {
    for (int i = 0; i < n; i++) a[i] = (double)(i % 7 + 1);
}



void naive_sgemm(int M,int N,int K,float alpha,float*A,float*B,float beta,float*C){
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++){
            float sum=0;
            for(int k=0;k<K;k++)
                sum+=A[i*K+k]*B[k*N+j];
            C[i*N+j]=alpha*sum+beta*C[i*N+j];
        }
}

void test_sgemm(){
    int M=3,N=3,K=3;
    float A[9],B[9],C1[9]={0},C2[9]={0};

    fillf(A,9); fillf(B,9);

    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
        M,N,K,1.0,A,K,B,N,0.0,C1,N);

    naive_sgemm(M,N,K,1.0,A,B,0.0,C2);

    printf("SGEMM: %s\n", cmpf(C1,C2,9)?"PASS":"FAIL");
}



void naive_dgemm(int M,int N,int K,double alpha,double*A,double*B,double beta,double*C){
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++){
            double sum=0;
            for(int k=0;k<K;k++)
                sum+=A[i*K+k]*B[k*N+j];
            C[i*N+j]=alpha*sum+beta*C[i*N+j];
        }
}

void test_dgemm(){
    int M=3,N=3,K=3;
    double A[9],B[9],C1[9]={0},C2[9]={0};

    filld(A,9); filld(B,9);

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
        M,N,K,1.0,A,K,B,N,0.0,C1,N);

    naive_dgemm(M,N,K,1.0,A,B,0.0,C2);

    printf("DGEMM: %s\n", cmpd(C1,C2,9)?"PASS":"FAIL");
}



void naive_ssyrk(int N,int K,float alpha,float*A,float beta,float*C){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float sum=0;
            for(int k=0;k<K;k++)
                sum+=A[i*K+k]*A[j*K+k];
            C[i*N+j]=alpha*sum+beta*C[i*N+j];
        }
}

void test_ssyrk(){
    int N=3,K=3;
    float A[9],C1[9]={0},C2[9]={0};

    fillf(A,9);

    cblas_ssyrk(CblasRowMajor,CblasUpper,CblasNoTrans,
        N,K,1.0,A,K,0.0,C1,N);

    naive_ssyrk(N,K,1.0,A,0.0,C2);

    printf("SSYRK: %s\n", cmpf_tri(C1,C2,N)?"PASS":"FAIL");
}



void naive_strmm(int N, float *A, float *B){
    float tmp[9]={0};

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            float sum = 0;
            for(int k=i;k<N;k++) 
                sum += A[i*N+k]*B[k*N+j];
            tmp[i*N+j] = sum;
        }
    }

    for(int i=0;i<N*N;i++) B[i]=tmp[i];
}

void test_strmm(){
    int N=3;
    float A[9]={1,0,0,2,3,0,4,5,6}; 
    float B1[9],B2[9];

    fillf(B1,9); fillf(B2,9);

    cblas_strmm(CblasRowMajor,CblasLeft,CblasUpper,
        CblasNoTrans,CblasNonUnit,
        N,N,1.0,A,N,B1,N);

    naive_strmm(N,A,B2);

    printf("STRMM: %s\n", cmpf(B1,B2,9)?"PASS":"FAIL");
}



void naive_strsm(int N,float*A,float*B){
    for(int i=N-1;i>=0;i--){
        for(int j=0;j<N;j++){
            for(int k=i+1;k<N;k++)
                B[i*N+j]-=A[i*N+k]*B[k*N+j];
            B[i*N+j]/=A[i*N+i];
        }
    }
}

void test_strsm(){
    int N=3;
    float A[9]={1,0,0,2,3,0,4,5,6}; 
    float B1[9],B2[9];

    fillf(B1,9); fillf(B2,9);

    cblas_strsm(CblasRowMajor,CblasLeft,CblasUpper,
        CblasNoTrans,CblasNonUnit,
        N,N,1.0,A,N,B1,N);

    naive_strsm(N,A,B2);

    printf("STRSM: %s\n", cmpf(B1,B2,9)?"PASS":"FAIL");
}



void naive_ssyr2k(int N,int K,float*A,float*B,float*C){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float sum=0;
            for(int k=0;k<K;k++)
                sum+=A[i*K+k]*B[j*K+k] + B[i*K+k]*A[j*K+k];
            C[i*N+j]+=sum;
        }
}

void test_ssyr2k(){
    int N=3,K=3;
    float A[9],B[9],C1[9]={0},C2[9]={0};

    fillf(A,9); fillf(B,9);

    cblas_ssyr2k(CblasRowMajor,CblasUpper,CblasNoTrans,
        N,K,1.0,A,K,B,K,0.0,C1,N);

    naive_ssyr2k(N,K,A,B,C2);

    printf("SSYR2K: %s\n", cmpf_tri(C1,C2,N)?"PASS":"FAIL");
}



int main() {
    test_sgemm();
    test_dgemm();
    test_ssyrk();
    test_strmm();
    test_strsm();
    test_ssyr2k();
    printf("All tests done.\n");
    return 0;
}