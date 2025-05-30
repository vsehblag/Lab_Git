#include <iostream>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <clocale>
#include <map>
#include <tuple>
#include <cmath>
#include <omp.h> 
#include "LOS.h"

// Сохранение решения в csv файл
void save_decision_to_csv(const std::string& filename, const std::vector<double>& rho, size_t x_cells, size_t z_cells) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Ошибка при открытии файла для записи rho!" << std::endl;
        return;
    }

    for (size_t i = 0; i < z_cells; ++i) {
        for (size_t j = 0; j < x_cells; ++j) {
            fout << rho[i * x_cells + j];
            if (j != x_cells - 1)
                fout << ",";
        }
        fout << "\n";
    }

    fout.close();
    std::cout << "Результат обратной задачи сохранён в: " << filename << std::endl;
}

// Вычисление вклада ячейки k в конкретный приемник
double compute_g_k_contribution(
    double x_receiver,
    double x_center, double z_center,
    double dx, double dz)
{

    const std::vector<double> gauss_pts = {
        -0.906179845938664, -0.538469310105683, 0.0,
         0.538469310105683, 0.906179845938664
    };

    const std::vector<double> gauss_wts = {
        0.236926885056189, 0.478628670499366,
        0.568888888888889, 0.478628670499366,
        0.236926885056189
    };

    double area_per_cell = dx * dz;
    double g_k_sum = 0.0;

    for (size_t i = 0; i < gauss_pts.size(); ++i) {
        for (size_t j = 0; j < gauss_pts.size(); ++j) {
            double xi = gauss_pts[i];
            double eta = gauss_pts[j];
            double weight = gauss_wts[i] * gauss_wts[j];

            double x_gauss = x_center + xi * (dx / 2.0);
            double z_gauss = z_center + eta * (dz / 2.0);

            double dx_r = x_receiver - x_gauss;
            double dz_r = z_gauss;
            double r = std::sqrt(dx_r * dx_r + dz_r * dz_r);

            if (r == 0.0) continue;

            double g_part = area_per_cell * z_gauss / (4.0 * PI * r * r * r);
            g_k_sum += weight * g_part;
        }
    }

    return g_k_sum;
}

// Вычисление матрицы вкладов ячеек в приемники
std::vector<std::vector<double>> compute_all_unitary_responses(
    const std::vector<double>& receiver_coords,
    const std::map<CellKey, CellData>& area_coords,
    double dx, double dz)
{
    size_t n_receivers = receiver_coords.size();
    size_t n_cells = area_coords.size();

    std::vector<std::vector<double>> G(n_cells, std::vector<double>(n_receivers, 0.0));

    size_t q = 0;
    for (const auto& [key, data] : area_coords) {
        double x_center = std::get<0>(data);
        double z_center = std::get<1>(data);

        for (size_t i = 0; i < n_receivers; ++i) {
            G[q][i] = compute_g_k_contribution(receiver_coords[i], x_center, z_center, dx, dz);
        }

        ++q;
    }

    return G;
}

// Вычисление вектора правой части
std::vector<double> compute_b_vector(
    const std::vector<double>& delta_g,
    const std::vector<std::vector<double>>& G)
{
    size_t n_cells = G.size();
    size_t n_receivers = delta_g.size();

    std::vector<double> b(n_cells, 0.0);

    for (size_t q = 0; q < n_cells; ++q) {
        double bq = 0.0;
        for (size_t i = 0; i < n_receivers; ++i) {
            bq += delta_g[i] * G[q][i];
        }
        b[q] = bq;
    }

    return b;
}

// Построение матрицы C
std::vector<std::vector<double>> build_C_dense_matrix(int x_cells, int z_cells, const std::vector<double>& gamma) {
    int N = x_cells * z_cells;
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    auto idx = [x_cells](int i, int j) { return i * x_cells + j; };

    for (int i = 0; i < z_cells; ++i) {
        for (int j = 0; j < x_cells; ++j) {
            int row = idx(i, j);
            double gamma_i = gamma[row];

            std::vector<int> neighbors;

            if (i > 0) neighbors.push_back(idx(i - 1, j));
            if (i < z_cells - 1) neighbors.push_back(idx(i + 1, j));
            if (j > 0) neighbors.push_back(idx(i, j - 1));
            if (j < x_cells - 1) neighbors.push_back(idx(i, j + 1));

            // Внедиагональные элементы
            for (int col : neighbors) {
                C[row][col] = -(gamma_i + gamma[col]);
            }

            // Диагональный элемент
            double sum_gamma_neighbors = 0.0;
            for (int col : neighbors) {
                sum_gamma_neighbors += gamma[col];
            }

            C[row][row] = gamma_i * neighbors.size() + sum_gamma_neighbors;
        }
    }

    return C;
}

// Добавление регуляризации в СЛАУ
std::vector<std::vector<double>> add_regularization(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& C,
    double alpha)
{
    int N = A.size();
    std::vector<std::vector<double>> result(N, std::vector<double>(N, 0.0));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = A[i][j] + C[i][j];
        }
        result[i][i] += alpha; // Добавляем alpha * I
    }

    return result;
}

// Вычисление значения функционала
double compute_functional(
    const std::vector<double>& x,            
    const Matrix& G,                     
    const std::vector<double>& delta_g,   
    double alpha,
    const std::vector<double>& gamma,
    size_t x_cells, size_t z_cells
) {
    const size_t N = delta_g.size();    // число приёмников
    const size_t K = x.size();          // число ячеек

    double J = 0.0;

    for (size_t i = 0; i < N; ++i) {
        double g_model = 0.0;
        for (size_t k = 0; k < K; ++k) {
            g_model += x[k] * G[k][i];
        }
        double residual = delta_g[i] - g_model;
        J += residual * residual;
    }

    for (size_t k = 0; k < K; ++k) {
        J += alpha * x[k] * x[k];
    }

    for (size_t i = 0; i < z_cells; ++i) {
        for (size_t j = 0; j < x_cells; ++j) {
            size_t idx = i * x_cells + j;
            double penalty = 0.0;
            if (j > 0)               penalty += std::pow(x[idx] - x[idx - 1], 2);
            if (j + 1 < x_cells)     penalty += std::pow(x[idx] - x[idx + 1], 2);
            if (i > 0)               penalty += std::pow(x[idx] - x[idx - x_cells], 2);
            if (i + 1 < z_cells)     penalty += std::pow(x[idx] - x[idx + x_cells], 2);
            J += gamma[idx] * penalty;
        }
    }

    return J;
}

// Решение обратной задачи с учетом параметров регуляризации
std::vector<double> solve_reverse_problem(
    std::vector<double>& gamma,
    const Matrix& A,                  
    const Matrix& G,                
    const std::vector<double>& delta_g,
    double alpha,
    size_t x_cells, size_t z_cells,
    double percent_threshold,                   // допустимый прирост функционала
    int max_iters                               // максимальное число итераций
) {
    int K = x_cells * z_cells;  // число ячеек (K)

    // Вычисляем вектор правой части СЛАУ
    auto b = compute_b_vector(delta_g, G);
    double J0 = std::numeric_limits<double>::max();  // первый функционал

    std::vector<double> final_X(K, 0.0);

    // Построение начальной матрицы C на основе gamma
    Matrix C = build_C_dense_matrix(x_cells, z_cells, gamma);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Формируем регуляризованную матрицу A_reg
        Matrix A_reg = add_regularization(A, C, alpha);

        // Конвертируем A_reg в разреженный формат для решения СЛАУ
        SparseMatrix sparse_A = convert_to_sparse(A_reg);

        // Решаем СЛАУ 
        std::vector<double> X(K, 0.0);

        calculate_LOS_diag(
            sparse_A.N, maxitr, eps,
            sparse_A.ig, sparse_A.jg, sparse_A.gg, sparse_A.di,
            b, X, false
        );

        // Вычисляем функционал 
        double J = compute_functional(X, G, delta_g, alpha, gamma, x_cells, z_cells);
        std::cout << "итерация " << iter << " J = " << J << std::endl;

        // Проверяем относительный прирост функционала
        if (iter == 0) {
            J0 = J;
        }
        else {
            double rel_change = std::abs(J - J0) / J0;
            if (rel_change > percent_threshold) {
                std::cout << "Относительное изменение функционала превысило "
                    << percent_threshold * 100 << "% — прекращаем\n";
                break;
            }
        }

        final_X = X;  

        // Вычисляем градиенты решения (по соседним ячейкам)
        std::vector<double> grad_X(K, 0.0);
        for (size_t j = 0; j < z_cells; ++j) {
            for (size_t i = 0; i < x_cells; ++i) {
                size_t idx = j * x_cells + i;
                double g = 0.0;
                if (i > 0)           g += std::abs(X[idx] - X[idx - 1]);
                if (i + 1 < x_cells) g += std::abs(X[idx] - X[idx + 1]);
                if (j > 0)           g += std::abs(X[idx] - X[idx - x_cells]);
                if (j + 1 < z_cells) g += std::abs(X[idx] - X[idx + x_cells]);
                grad_X[idx] = g;
            }
        }

        // Увеличиваем гамма в ячейках с большим градиентом решения

        double max_grad = *std::max_element(grad_X.begin(), grad_X.end());

        for (size_t idx = 0; idx < gamma.size(); ++idx) {
            double rel_grad = grad_X[idx] / (max_grad + 1e-12);
            double factor = 1.0 + rel_grad * 5;
            gamma[idx] *= factor;
        }

        // Перестраиваем матрицу C с обновлённым гамма
        C = build_C_dense_matrix(x_cells, z_cells, gamma);

    }

    return final_X;
}

// Вычисление поля в приемниках (прямая задача)
std::vector<double> calculate_delta_g_on_receivers(
    const std::vector<double>& receiver_coords,
    const std::map<CellKey, CellData>& area_coords,
    double dx, double dz)
{
    std::vector<double> results(receiver_coords.size(), 0.0);

    for (size_t rcv_idx = 0; rcv_idx < receiver_coords.size(); ++rcv_idx) {
        double x_receiver = receiver_coords[rcv_idx];
        double total_g = 0.0;

        for (const auto& [key, data] : area_coords) {
            double x_center = std::get<0>(data);
            double z_center = std::get<1>(data);
            float value = std::get<2>(data);

            if (value == 0.0f) continue;

            double g_k_sum = compute_g_k_contribution(x_receiver, x_center, z_center, dx, dz);

            total_g += value * g_k_sum;
        }

        results[rcv_idx] = total_g;
    }

    return results;
}

// Вычисление (x_center, z_center, value) для каждой подобласти
std::map<CellKey, CellData> calculate_area(
    const std::vector<float>& area,
    double x_start, double x_end,
    double z_start, double z_end,
    size_t x_cells, size_t z_cells)
{
    std::map<CellKey, CellData> grid;

    double dx = (x_end - x_start) / static_cast<double>(x_cells);
    double dz = (z_end - z_start) / static_cast<double>(z_cells);

    for (size_t i = 0; i < z_cells; ++i) {
        for (size_t j = 0; j < x_cells; ++j) {
            double x_center = x_start + j * dx + dx / 2.0;
            double z_center = z_start + i * dz + dz / 2.0;

            float value = area[i * x_cells + j];

            grid[{i, j}] = CellData(x_center, z_center, value);
        }
    }

    return grid;
}

// Вычисление позиций приемников на поверхности
std::vector<double> calculate_receivers_coords(double x_start, double x_finish, size_t count_receivers) {
    std::vector<double> coords;

    if (count_receivers == 0) {
        return coords;
    }

    coords.resize(count_receivers);

    if (1 == count_receivers) {
        coords[0] = x_start;
        return coords;
    }

    double step = (x_finish - x_start) / static_cast<double>(count_receivers - 1);

    for (size_t i = 0; i < count_receivers; ++i) {
        coords[i] = x_start + i * step;
    }

    return coords;
}

int main() {
    std::setlocale(LC_ALL, "Russian");
    std::ifstream fin("config.txt");
    if (!fin) {
        std::cerr << "Не удалось открыть файл!" << std::endl;
        return 1;
    }

    // Чтение X и Z (размеры области)
    double x_start_area, x_end_area;
    double z_start_area, z_end_area;
    fin >> x_start_area >> x_end_area;
    fin >> z_start_area >> z_end_area;

    // Чтение x и z (количество разбиений)
    size_t x, z;
    fin >> x >> z;

    // Чтение x*z значений искомого параметра в области
    std::vector<float> area(x * z);
    for (size_t i = 0; i < x * z; ++i) {
        fin >> area[i];
    }

    // Чтение x_start, x_finish, count_receivers
    double x_start, x_finish;
    size_t count_receivers;
    fin >> x_start >> x_finish >> count_receivers;
    
    // Чтение x_calculate_start_area и x_calculate_end_area для обратной задачи
    double x_calculate_start_area, x_calculate_end_area;
    fin >> x_calculate_start_area >> x_calculate_end_area;
    

    // Чтение z_calculate_start_area и z_calculate_end_area для обратной задачи
    double z_calculate_start_area, z_calculate_end_area;
    fin >> z_calculate_start_area >> z_calculate_end_area;

    // Чтение x_calculate и z_calculate для обратной задачи
    size_t x_calculate, z_calculate;
    fin >> x_calculate >> z_calculate;
    
    // Вычисление позиций приемников на поверхности
    std::vector<double> receiver_coords = calculate_receivers_coords(x_start, x_finish, count_receivers);

    auto area_coords = calculate_area(area, x_start_area, x_end_area, z_start_area, z_end_area, x, z);

    // Вычисляем площадь одной ячейки
    double dx = (x_end_area - x_start_area) / static_cast<double>(x);
    double dz = (z_end_area - z_start_area) / static_cast<double>(z);
    double area_per_cell = dx * dz;

    std::vector<double> delta_g = calculate_delta_g_on_receivers(receiver_coords, area_coords, dx, dz);

    std::vector<float> area_inverse(x_calculate * z_calculate, 0.0f);

    auto area_coords_inverse = calculate_area(area_inverse,
        x_calculate_start_area, x_calculate_end_area,
        z_calculate_start_area, z_calculate_end_area,
        x_calculate, z_calculate);

    double dx_invrese = (x_calculate_end_area - x_calculate_start_area) / static_cast<double>(x_calculate);
    double dz_invrese = (z_calculate_end_area - z_calculate_start_area) / static_cast<double>(z_calculate);

    auto G_unit = compute_all_unitary_responses(receiver_coords, area_coords_inverse, dx_invrese, dz_invrese);

    auto b = compute_b_vector(delta_g, G_unit);

    auto A = compute_matrix_A(G_unit);
    
    // Параметры регуляризации
    double alpha = 0.000005;
    std::vector<double> gamma(x_calculate * z_calculate, 1e-12);

    // Строим матрицу C
    auto C_dense = build_C_dense_matrix(x_calculate, z_calculate, gamma);

    // Складываем A + alpha * I + C
    auto A_reg_dense = add_regularization(A, C_dense, alpha);

    // Переводим матрицу в разреженный формат
    auto sparse_A = convert_to_sparse(A_reg_dense);

    // Начальное приближение X0 = 0
    std::vector<double> X(sparse_A.N, 0.0);
    
    // Решение обратной задачи
    std::vector<double> rho = solve_reverse_problem(
        gamma,
        A,               
        G_unit,          
        delta_g,
        alpha,
        x_calculate, z_calculate,
        0.05,
        1000
    );
    
    // Сохранение финального результата
    save_decision_to_csv("result.csv", rho, x_calculate, z_calculate);

    fin.close();
    return 0;
}
