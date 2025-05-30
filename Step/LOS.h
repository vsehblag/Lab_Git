#ifndef LOS_SOLVER_H
#define LOS_SOLVER_H

#include <vector>

constexpr double PI = 3.14159265358979323846;
using CellKey = std::pair<size_t, size_t>; // Ключ (i, j)
using CellData = std::tuple<double, double, float>; // (x_center, z_center, value)
using Matrix = std::vector<std::vector<double>>;
int maxitr = 10000;
double eps = 1e-12;

struct SparseMatrix {
    int N;                              // Размерность матрицы
    std::vector<double> di;             // Диагональные элементы
    std::vector<int> ig;                // Индексы начала строк в gg и jg
    std::vector<int> jg;                // Столбцы для внедиагональных элементов
    std::vector<double> gg;             // Внедиагональные элементы
};

// Используем пространство имен std
using std::vector;

// Число потоков
extern int threadNum;

// Сложение матриц в разреженном формате
void summation_matrices(vector<double>& B_di, vector<double>& B_gg,
    vector<double>& C_di, vector<double>& C_gg,
    double w, vector<double>& A_di, vector<double>& A_gg,
    int n);

// Скалярное произведение двух векторов
double sk_mult(vector<double>& a, vector<double>& b);

// Умножение разреженной матрицы на вектор
void mul_matr_to_vec(int& n, vector<int>& ig, vector<int>& jg,
    vector<double>& gg, vector<double>& di,
    vector<double>& x, vector<double>& y);

// Диагональное предобуславливание: вычисление вектора q = M * r
vector<double> calc_q(vector<double> M, vector<double> r, int n);

// Вычисление обратных значений диагональных элементов
vector<double> calc_inverse(const vector<double>& di, int n);

// Основной алгоритм LOS с диагональным предобуславливанием
void calculate_LOS_diag(int& n, int& maxitr, double& eps,
    vector<int>& ig, vector<int>& jg,
    vector<double>& gg, vector<double>& di,
    vector<double>& pr, vector<double>& X,
    bool isPrint);

// Сохранение решения прямой задачи в csv файл
void save_field_to_csv(const std::string& filename,
    const std::vector<double>& receiver_coords,
    const std::vector<double>& delta_g)
{
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Ошибка открытия файла для записи: " << filename << std::endl;
        return;
    }

    fout << "X,Delta_g\n";
    for (size_t i = 0; i < receiver_coords.size(); ++i) {
        fout << receiver_coords[i] << "," << delta_g[i] << "\n";
    }

    fout.close();
    std::cout << "Данные поля успешно сохранены в файл: " << filename << std::endl;
}

// Сложение матриц
Matrix compute_matrix_A(const Matrix& unit_responses)
{
    size_t num_cells = unit_responses.size();
    size_t num_receivers = unit_responses[0].size();

    Matrix A(num_cells, std::vector<double>(num_cells, 0.0));

    for (size_t q = 0; q < num_cells; ++q) {
        for (size_t s = q; s < num_cells; ++s) {
            double sum = 0.0;
            for (size_t i = 0; i < num_receivers; ++i) {
                sum += unit_responses[q][i] * unit_responses[s][i];
            }
            A[q][s] = sum;
            A[s][q] = sum; // матрица симметрична
        }
    }

    return A;
}

// Конвертация в разреженный формат
SparseMatrix convert_to_sparse(const std::vector<std::vector<double>>& A) {
    const int N = A.size();
    SparseMatrix sparse;
    sparse.N = N;
    sparse.di.resize(N);
    sparse.ig.resize(N + 1, -1);

    std::vector<double> gg_tmp;
    std::vector<int> jg_tmp;

    int index = 0;
    for (int i = 0; i < N; ++i) {
        bool row_started = false;

        for (int j = 0; j <= i; ++j) {
            double val = A[i][j];

            if (j == i) {
                sparse.di[i] = val;
            }
            else if (val != 0.0) {
                if (!row_started) {
                    sparse.ig[i] = index;
                    row_started = true;
                }

                gg_tmp.push_back(val);
                jg_tmp.push_back(j);
                ++index;
            }
        }
    }

    sparse.ig[N] = index;
    for (int i = N; i >= 0; --i) {
        if (sparse.ig[i] == -1)
            sparse.ig[i] = sparse.ig[i + 1];
    }

    sparse.gg = std::move(gg_tmp);
    sparse.jg = std::move(jg_tmp);

    return sparse;
}

// Конвертация в плотный формат
Matrix convert_to_dense(const SparseMatrix& sparse) {
    int N = sparse.N;
    Matrix A(N, std::vector<double>(N, 0.0));

    // Восстановление диагонали
    for (int i = 0; i < N; ++i) {
        A[i][i] = sparse.di[i];
    }

    // Восстановление нижней и верхней треугольной частей
    for (int i = 0; i < N; ++i) {
        for (int k = sparse.ig[i]; k < sparse.ig[i + 1]; ++k) {
            int j = sparse.jg[k];
            double val = sparse.gg[k];
            A[i][j] = val;
            A[j][i] = val;  // симметрично
        }
    }

    return A;
}

#endif // LOS_SOLVER_H