#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;
using namespace chrono;

const double EPS = 1e-12;

class Matrix {
public:
    vector<vector<double>> data;
    int rows, cols;

    Matrix(int n, int m) : rows(n), cols(m), data(n, vector<double>(m, 0)) {}
};

// Генерация матрицы A по варианту 9
Matrix generate_A(int N) {
    Matrix A(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                A.data[i][j] = 100.0;
            }
            else {
                A.data[i][j] = 1.0 + 0.3 * (i + 1) - 0.1 * (j + 1);
            }
        }
    }
    return A;
}

// Генерация вектора f = A * x*, где x* = [1,1,...,1]
vector<double> generate_f(const Matrix& A) {
    int N = A.rows;
    vector<double> f(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            f[i] += A.data[i][j] * 1.0;
        }
    }
    return f;
}

// LU-разложение с выбором ведущего элемента
void LU_decomposition(Matrix& A, vector<int>& perm) {
    int N = A.rows;
    perm.resize(N);
    for (int i = 0; i < N; ++i) perm[i] = i;

    for (int k = 0; k < N; ++k) {
        // Поиск ведущего элемента
        int max_row = k;
        for (int i = k; i < N; ++i) {
            if (abs(A.data[i][k]) > abs(A.data[max_row][k])) {
                max_row = i;
            }
        }
        swap(A.data[k], A.data[max_row]);
        swap(perm[k], perm[max_row]);

        // Исключение
        for (int i = k + 1; i < N; ++i) {
            A.data[i][k] /= A.data[k][k];
            for (int j = k + 1; j < N; ++j) {
                A.data[i][j] -= A.data[i][k] * A.data[k][j];
            }
        }
    }
}

// Решение СЛАУ методом LU
vector<double> LU_solve(Matrix& LU, const vector<int>& perm, const vector<double>& f) {
    int N = LU.rows;
    vector<double> x(N), y(N);

    // Прямой ход
    for (int i = 0; i < N; ++i) {
        y[i] = f[perm[i]];
        for (int j = 0; j < i; ++j) {
            y[i] -= LU.data[i][j] * y[j];
        }
    }

    // Обратный ход
    for (int i = N - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < N; ++j) {
            x[i] -= LU.data[i][j] * x[j];
        }
        x[i] /= LU.data[i][i];
    }
    return x;
}

// QR-разложение методом Гивенса
void QR_decomposition(const Matrix& A, Matrix& Q, Matrix& R) {
    int N = A.rows;
    Q = Matrix(N, N);
    R = A;

    for (int i = 0; i < N; ++i) Q.data[i][i] = 1.0;

    for (int j = 0; j < N; ++j) {
        for (int i = N - 1; i > j; --i) {
            if (abs(R.data[i][j]) < EPS) continue;

            double a = R.data[j][j];
            double b = R.data[i][j];
            double norm = sqrt(a * a + b * b);
            double c = a / norm;
            double s = b / norm;

            // Обновляем строки R
            for (int k = j; k < N; ++k) {
                double Rj = R.data[j][k];
                double Ri = R.data[i][k];
                R.data[j][k] = c * Rj + s * Ri;
                R.data[i][k] = -s * Rj + c * Ri;
            }

            // Обновляем матрицу Q
            for (int k = 0; k < N; ++k) {
                double Qk = Q.data[k][j];
                double Qi = Q.data[k][i];
                Q.data[k][j] = c * Qk + s * Qi;
                Q.data[k][i] = -s * Qk + c * Qi;
            }
        }
    }
}

// Решение СЛАУ методом QR
vector<double> QR_solve(const Matrix& Q, const Matrix& R, const vector<double>& f) {
    int N = R.rows;
    // Вычисляем Q^T * f
    vector<double> qt_f(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            qt_f[i] += Q.data[j][i] * f[j];
        }
    }

    // Обратная подстановка для R
    vector<double> x(N);
    for (int i = N - 1; i >= 0; --i) {
        x[i] = qt_f[i];
        for (int j = i + 1; j < N; ++j) {
            x[i] -= R.data[i][j] * x[j];
        }
        x[i] /= R.data[i][i];
    }
    return x;
}// Вычисление нормы вектора
double vector_norm(const vector<double>& v) {
    double norm = 0.0;
    for (double x : v) norm += x * x;
    return sqrt(norm);
}

int main() {
    setlocale(LC_ALL, "Russian");
    vector<int> sizes = { 250, 500, 1000 };
    const int runs = 3;

    for (int N : sizes) {
        cout << "N = " << N << endl;

        long long total_lu_time = 0, total_qr_time = 0;
        double total_lu_err = 0.0, total_qr_err = 0.0;

        for (int run = 0; run < runs; ++run) {
            // Генерация данных
            Matrix A = generate_A(N);
            vector<double> f = generate_f(A);

            // LU-метод
            Matrix LU = A;
            vector<int> perm;
            auto start = high_resolution_clock::now();
            LU_decomposition(LU, perm);
            vector<double> x_lu = LU_solve(LU, perm, f);
            auto stop = high_resolution_clock::now();
            auto lu_time = duration_cast<milliseconds>(stop - start).count();
            total_lu_time += lu_time;

            // QR-метод
            Matrix Q(N, N), R(N, N);
            start = high_resolution_clock::now();
            QR_decomposition(A, Q, R);
            vector<double> x_qr = QR_solve(Q, R, f);
            stop = high_resolution_clock::now();
            auto qr_time = duration_cast<milliseconds>(stop - start).count();
            total_qr_time += qr_time;

            // Вычисление погрешностей
            double lu_err = 0.0, qr_err = 0.0;
            for (int i = 0; i < N; ++i) {
                lu_err += (x_lu[i] - 1.0) * (x_lu[i] - 1.0);
                qr_err += (x_qr[i] - 1.0) * (x_qr[i] - 1.0);
            }
            lu_err = sqrt(lu_err) / sqrt(N);
            qr_err = sqrt(qr_err) / sqrt(N);

            total_lu_err += lu_err;
            total_qr_err += qr_err;
        }

        cout << "LU среднее время за 10 запусков: " << total_lu_time / runs <<  " ms\tСредняя погрешность за 10 запусков: " << total_lu_err / runs << endl;
        cout << "QR среднее время за 10 запусков: " << total_qr_time / runs <<  " ms\tСредняя погрешность за 10 запусков: " << total_qr_err / runs << endl << endl;
    }

    return 0;
}