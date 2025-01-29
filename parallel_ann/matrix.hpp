#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <math.h>

using namespace std;

    template<typename T>class Matrix
    {
    public:
        int _cols;
        int _rows;
        vector<T> _vals;


    public:
        Matrix(int cols, int rows)
            : _cols(cols),
            _rows(rows),
            _vals({})
        {
            _vals.resize(rows * cols, T());
        }

        Matrix()
             : _cols(0),
            _rows(0),
            _vals({})
        {
        }

        T& at(int col, int row)
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
            for (int y = 0; y < output._rows; y++)
                for (int x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = -at(x, y);
                }
            return output;
        }


        Matrix multiply(Matrix& target)
        {
            assert(_cols == target._rows);
            Matrix output(target._cols, _rows);
            for (int y = 0; y < output._rows; y++)
                for (int x = 0; x < output._cols; x++)
                {
                    T result = T();
                    for (int k = 0; k < _cols; k++)
                        result += at(k, y) * target.at(x, k);
                    output.at(x, y) = result;
                }
            return output;
        }

        Matrix multiplyElements(Matrix& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix output(_cols, _rows);
            for (int y = 0; y < output._rows; y++)
                for (int x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) * target.at(x, y);
                }
            return output;
        }

        float sumElements()
        {
            float sum = 0;
            for (int y = 0; y < _rows; y++)
                for (int x = 0; x < _cols; x++)
                {
                    sum += at(x, y);
                }
            return sum;
        }

        float max()
        {
            float max = at(0,0);
            for (int y = 0; y < _rows; y++)
                for (int x = 0; x < _cols; x++)
                {
                    max = ((max <= at(x, y))? at(x, y) : max);
                }
            return max;
        }

        Matrix logSoftmax(){
    
            float max = this -> max();

            cout << "max: " << max << endl;

            Matrix<float> targetMatrix = this -> addScaler(-max);

            cout << "unactivated after normalisation:" << endl;
            targetMatrix.print();

            targetMatrix = targetMatrix.applyFunction([](const float &val){
                        return exp(val);
                    });

            cout << "output after exp:" << endl;
            targetMatrix.print();

            float sum = targetMatrix.sumElements();
            sum = log(sum);

            targetMatrix = targetMatrix.addScaler(-sum);

            cout << "output after sum:" << endl;
            targetMatrix.print();

            targetMatrix = targetMatrix.addScaler(-max);

            cout << "output after logsoftmax:" << endl;
            targetMatrix.print();

            return targetMatrix;
        }

        Matrix Softmax(){
            
            Matrix targetMatrix(_cols, _rows);
            for (int Y = 0; Y < _rows; Y++){
                for (int X = 0; X < _cols; X++){

                    Matrix output(_cols, _rows);
                    float target = at(X, Y);

                    for (int y = 0; y < output._rows; y++){
                        for (int x = 0; x < output._cols; x++){
                            output.at(x, y) = at(x, y) - target;
                        }
                    }
                    
                    output = output.applyFunction([](const float &val){
                                return exp(val);
                            });

                    float sum = output.sumElements();
                    targetMatrix.at(X, Y) = 1/sum;
                }
            }
    
            return targetMatrix;
        }

        void print(){
            for (int y = 0; y < _rows; y++){
                for (int x = 0; x < _cols; x++)
                {
                    printf("%f ", at(x, y));
                }
                printf("\n");
            }
        }

        Matrix add(Matrix& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix output(_cols, _rows);
            for (int y = 0; y < output._rows; y++)
                for (int x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + target.at(x, y);
                }
            return output;
        }

        Matrix addLimited(Matrix& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix output(_cols, _rows);
            for (int y = 0; y < output._rows; y++)
                for (int x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + target.at(x, y);
                    
                    if(output.at(x, y) > 1)
                        output.at(x, y) = 1;


                    if(output.at(x, y) < 0)
                        output.at(x, y) = 0;
                }
            return output;
        }

        //function<[return type of function](type arg1, type arg2...)> func LAMBDA FUNCTIONS
        Matrix applyFunction(function<T(const T&)> func)
        {
            Matrix output(_cols, _rows);
            for (int y = 0; y < output._rows; y++)
                for (int x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = func(at(x, y));
                }
            return output;
        }

        Matrix multiplyScaler(float s)
        {
            Matrix output(_cols, _rows);
            for (int y = 0; y < output._rows; y++)
                for (int x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) * s;
                }
            return output;

        }

        Matrix addScaler(float s)
        {
            Matrix output(_cols, _rows);
            for (int y = 0; y < output._rows; y++)
                for (int x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + s;
                }
            return output;

        }
        Matrix transpose()
        {
            Matrix output(_rows, _cols);
            for (int y = 0; y < _rows; y++)
                for (int x = 0; x < _cols; x++)
                {
                    output.at(y, x) = at(x, y);
                }
            return output;
        }

        Matrix cofactor(int col, int row)
        {
            Matrix output(_cols - 1, _rows - 1);
            int i = 0;
            for (int y = 0; y < _rows; y++)
                for (int x = 0; x < _cols; x++)
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
                for (int x = 0; x < _cols; x++)
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
            for (int y = 0; y < _rows; y++)
                for (int x = 0; x < _cols; x++)
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
            for (int y = 0; y < adj._cols; y++)
                for (int x = 0; x < adj._rows; x++)
                {
                    adj.at(x, y) = adj.at(x, y) / factor;
                }
            return adj;
        }



    }; // class Matrix

    template<typename T> void LogMatrix(Matrix<T>& mat)
    {
        for (int y = 0; y < mat._rows; y++)
        {
            for (int x = 0; x < mat._cols; x++)
                cout << setw(10) << mat.at(x, y) << " ";
            cout << endl;
        }
    }



