#include <stdio.h> 
#include <locale.h> 
#include <math.h> 
#include <vector> 
#include <time.h> 
#include <omp.h> 
#include <iostream> 
//------------------------------- 
using namespace std;
int threadNum = 12;
//-------------------------------
void summation_matrices(vector<double>& B_di, vector<double>& B_gg,
    vector<double>& C_di, vector<double>& C_gg,
    double w, vector<double>& A_di, vector<double>& A_gg, int n) {

    A_di.clear();
    A_gg.clear();

    A_di = B_di;
    for (int i = 0; i < n; i++) {
        A_di[i] += w * C_di[i];
    }

    A_gg = B_gg;
    for (int i = 0; i < B_gg.size(); i++) {
        A_gg[i] += w * C_gg[i];
    }
}
//-------------------------------
double sk_mult(vector <double>& a, vector <double>& b) {
    double mul = 0;
#pragma omp parallel for reduction(+:mul)
    for (int i = 0; i < a.size(); i++)
        mul += a[i] * b[i];
    return mul;
}
//-------------------------------
void mul_matr_to_vec(int& n, vector <int>& ig, vector <int>& jg, vector <double>& gg,
    vector <double>& di, vector <double>& x, vector <double>& y)
{
    int i, j, k;
    int rank;
    int adr;

    std::vector<double> y_omp(threadNum * n, 0.0);

#pragma omp parallel shared(ig, jg, gg, di, x, y, y_omp) private(i, j, k, rank, adr) num_threads(threadNum)
    {
#pragma omp for  
        for (i = 0; i < n; i++)
            y[i] = 0.0;
#pragma omp for  
        for (i = 0; i < n; i++)
        {
            rank = omp_get_thread_num();
            if (rank == 0)
            {
                y[i] = di[i] * x[i];
                for (j = ig[i]; j <= ig[i + 1] - 1; j++)
                {
                    k = jg[j];
                    y[i] += gg[j] * x[k];
                    y[k] += gg[j] * x[i];
                }
            }
            else
            {
                adr = (rank - 1) * n;
                y_omp[adr + i] = di[i] * x[i];
                for (j = ig[i]; j <= ig[i + 1] - 1; j++)
                {
                    k = jg[j];
                    y_omp[adr + i] += gg[j] * x[k];
                    y_omp[adr + k] += gg[j] * x[i];
                }
            }
        }
#pragma omp for 
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < threadNum - 1; j++)
                y[i] += y_omp[j * n + i];
        }
    }
}

//-------------------------------
// Диагональное предобуславливание
vector<double> calc_q(vector<double> M, vector<double> r, int n) {
    int i;
    vector<double>q(n);
#pragma omp parallel for shared(M, r, q) private(i)
    for (i = 0; i < n; i++) {
        q[i] = r[i] * M[i];
    }
    return q;
}
//-------------------------------
vector<double> calc_inverse(const vector<double>& di, int n) {
    vector<double> di_inverse(n);

    for (int i = 0; i < n; i++) {
        di_inverse[i] = 1.0 / di[i];
    }

    return di_inverse;
}
//-------------------------------
void calculate_LOS_diag(int& n, int& maxitr, double& eps, vector <int>& ig, vector
    <int>& jg, vector <double>& gg, vector <double>& di, vector <double>& pr, vector
    <double>& X, bool isPrint)
{
    vector <double> a, r, z, p, s, w, Ax;
    a.resize(n);
    r.resize(n);
    z.resize(n);
    p.resize(n);
    s.resize(n);
    w.resize(n);
    Ax.resize(n);

    int i = 0;
    int iter = 0;
    double alpha, b, sk, nev = 0.0;
    // Вычисляем квадрат нормы правой части
    double pr_norma = sk_mult(pr, pr);
    double eps_pr = eps * pr_norma;

    // Вычисляем вектор обратный диагональным элементам
    vector<double> di_inverse = calc_inverse(di, n);

    // Вычисляем начальное значение A * x0
    mul_matr_to_vec(n, ig, jg, gg, di, X, Ax);

    // Вычиляем начальное приближение r0
    for (int i = 0; i < n; i++)
        r[i] = pr[i] - Ax[i];

    // Вычисляем начальные значения p0 и s0
    p = calc_q(di_inverse, r, n);
    s = p;

    // Вычисляем начальные значения a0 и z0
    mul_matr_to_vec(n, ig, jg, gg, di, p, a);
    z = a;

    // Вычисляем начальное значениe w0
    w = calc_q(di_inverse, z, n);

    // Вычислим невязку
    for (i = 0; i < n; i++)
        nev += r[i] * r[i];

    if (isPrint) printf_s("Первая невязка %.10le\n", sqrt(nev / pr_norma));

    for (iter = 0; iter < maxitr && nev > eps_pr; iter++)
    {
        sk = sk_mult(w, z);
        if (sk == 0)
            nev = 0;
        else
        {
            alpha = sk_mult(w, r) / sk;
#pragma omp parallel for private(i)
            for (i = 0; i < n; i++)
            {
                X[i] += alpha * p[i];
                r[i] -= alpha * z[i];
                s[i] -= alpha * w[i];
            }

            mul_matr_to_vec(n, ig, jg, gg, di, s, a);

            b = -1 * sk_mult(w, a) / sk;

#pragma omp parallel for private(i)
            for (i = 0; i < n; i++)
            {
                p[i] = s[i] + b * p[i];
                z[i] = a[i] + b * z[i];
            }

            w = calc_q(di_inverse, z, n);


            nev = sk_mult(r, r);

            if (isPrint && (0 != iter && iter % 1000 == 0) || nev <= eps_pr || iter + 1 >= maxitr)
                printf_s("Итерация №%d\nНевязка - %.10le \n\n", iter + 1, sqrt(nev / pr_norma));
        }
    }
}