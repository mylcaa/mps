#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>

using namespace std;


namespace sp
{
    template<typename T>class Matrix
    {
    public:
        uint32_t _cols;
        uint32_t _rows;
        vector<T> _vals;


    public:
        Matrix(uint32_t cols, uint32_t rows)
            : _cols(cols),
            _rows(rows),
            _vals({})
        {
            vals.resize(rows * cols, T());
        }

        Matrix()
             : _cols(0),
            _rows(0),
            _vals({})
        {
        }

        T& at(uint32_t col, uint32_t row)
        {
            return _vals[row * _cols + col];
        }

        bool isSquare()
        {
            return _rows == _cols;
        }


        Matrix negative()
        {
            Matrix output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = -at(x, y);
                }
            return output;
        }


        Matrix multiply(Matrix& target)
        {
            assert(_cols == target._rows);
            Matrix output(target._cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    T result = T();
                    for (uint32_t k = 0; k < _cols; k++)
                        result += at(k, y) * target.at(x, k);
                    output.at(x, y) = result;
                }
            return output;
        }

        Matrix multiplyElements(Matrix& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) * target.at(x, y);
                }
            return output;
        }

        float sumElements()
        {
            float sum = 0;
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    sum += at(x, y);
                }
            return sum;
        }


        Matrix add(Matrix& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + target.at(x, y);
                }
            return output;
        }

        //function<[return type of function](type arg1, type arg2...)> func LAMBDA FUNCTIONS
        Matrix applyFunction(function<T(const T&)> func)
        {
            Matrix output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = func(at(x, y));
                }
            return output;
        }

        Matrix multiplyScaler(float s)
        {
            Matrix output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) * s;
                }
            return output;

        }

        Matrix addScaler(float s)
        {
            Matrix output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + s;
                }
            return output;

        }
        Matrix transpose()
        {
            Matrix output(_rows, _cols);
            for (uint32_t y = 0; y < _rows; y++)
                for (uint32_t x = 0; x < _cols; x++)
                {
                    output.at(y, x) = at(x, y);
                }
            return output;
        }

        Matrix cofactor(uint32_t col, uint32_t row)
        {
            Matrix output(_cols - 1, _rows - 1);
            uint32_t i = 0;
            for (uint32_t y = 0; y < _rows; y++)
                for (uint32_t x = 0; x < _cols; x++)
                {
                    if (x == col || y == row) continue;
                    output._vals[i++] = at(x, y);
                }

            return output;
        }

        T determinant()
        {
            assert(_rows == _cols);
            T output = T();
            if (_rows == 1)
            {
                return _vals[0];
            }
            else
            {
                int32_t sign = 1;
                for (uint32_t x = 0; x < _cols; x++)
                {
                    output += sign * at(x, 0) * cofactor(x, 0).determinant();
                    sign *= -1;
                }
            }

            return output;
        }

        Matrix adjoint()
        {
            assert(_rows == _cols);
            Matrix output(_cols, _rows);
            int32_t sign = 1;
            for (uint32_t y = 0; y < _rows; y++)
                for (uint32_t x = 0; x < _cols; x++)
                {
                    output.at(x, y) = sign * cofactor(x, y).determinant();
                    sign *= -1;
                }
            output = output.transpose();

            return output;
        }

        Matrix inverse()
        {
            Matrix adj = adjoint();
            T factor = determinant();
            for (uint32_t y = 0; y < adj._cols; y++)
                for (uint32_t x = 0; x < adj._rows; x++)
                {
                    adj.at(x, y) = adj.at(x, y) / factor;
                }
            return adj;
        }



    }; // class Matrix

    template<typename T> void LogMatrix(Matrix<T>& mat)
    {
        for (uint32_t y = 0; y < mat._rows; y++)
        {
            for (uint32_t x = 0; x < mat._cols; x++)
                cout << setw(10) << mat.at(x, y) << " ";
            cout << endl;
        }
    }

}