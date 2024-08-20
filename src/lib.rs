#![allow(dead_code)]

use std::ops::{
    Add,
    Mul,
    Neg,
    Index,
};
use std::error::Error;
use std::fmt::{Display, Debug, Formatter, Error as FmtError};
use num::Num;

#[derive(Debug, PartialEq, Clone)]
struct Matrix<T> {
    rows: usize,
    columns: usize,
    grid: Vec<Vec<T>>,
}

#[derive(Debug, PartialEq)]
enum MatrixError {
    EmptyMatrixError,
    RowsNotOfSameSizeError,
    NotTheSameSizeError,
    IncompatibleSizeMulError,
    MatrixCantBeZeroSizedError,
    NotSquaredMatrixError,
    MatrixTooSmall,
    InvalidVecSize,
}

impl Display for MatrixError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::EmptyMatrixError =>
                write!(f, "Matrix is empty."),
            Self::RowsNotOfSameSizeError =>
                write!(f, "The rows are not of the same size."),
            Self::NotTheSameSizeError =>
                write!(f, "The Matrixes have different size."),
            Self::IncompatibleSizeMulError =>
                write!(f, "The Matrixes are not compatible."),
            Self::MatrixCantBeZeroSizedError =>
                write!(f, "Matrix can't be of zero size."),
            Self::NotSquaredMatrixError =>
                write!(f, "Matrix is not squared."),
            Self::MatrixTooSmall =>
                write!(f, "Matrix is too small."),
            Self::InvalidVecSize =>
                write!(f, "Vec of invalid size."),
        }
    }
}

impl Error for MatrixError {}

impl<T> Matrix<T> {
    fn new(grid: Vec<Vec<T>>) -> Result<Self, MatrixError> {
        if grid.len() == 0 {
            return Err(MatrixError::EmptyMatrixError);
        }

        if !are_rows_same_size(&grid) {
            return Err(MatrixError::RowsNotOfSameSizeError);
        }

        let rows = grid.len();
        let columns = grid[0].len();

        Ok(Self {
            rows,
            columns,
            grid,
        })
    }

    fn from_vec(
        v: Vec<T>, rows: usize
    ) -> Result<Self, MatrixError> {
        let len: f32 = v.len() as f32;
        let x_f32: f32 = rows as f32;
        if (len / x_f32) % 1. != 0. {
            return Err(MatrixError::InvalidVecSize);
        }

        let mut result: Vec<Vec<T>> = Vec::new();

        let mut i = 0;
        let mut sub_result: Vec<T> = Vec::new();
        for elem in v {
            sub_result.push(elem);
            i+=1;
            if i == rows {
                result.push(sub_result);
                sub_result = Vec::new();
                i = 0;
            }
        }

        Ok(Matrix::new(result).unwrap())
    }

    fn get_nb_rows(&self) -> usize {
        self.rows
    }

    fn get_nb_columns(&self) -> usize {
        self.columns
    }

    fn get_row(&self, index: usize) -> Option<Vec<&T>> {
        if index >= self.rows {
            None
        }
        else {
            Some(self.grid[index].iter().map(|e| e).collect::<Vec<&T>>())
        }
    }

    fn get_column(&self, index: usize) -> Option<Vec<&T>> {
        if index >= self.columns {
            return None;
        }

        let mut column: Vec<&T> = Vec::new();

        for x in 0..self.rows {
            column.push(&self.grid[x][index]);
        }

        Some(column)
    }

    fn is_squared(&self) -> bool {
        self.rows == self.columns
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = MatrixIterator<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        MatrixIterator::new(self)
    }
}

struct MatrixIterator<T> {
    matrix_values: Vec<T>,
}

impl<T> MatrixIterator<T> {
    fn new(matrix: Matrix<T>) -> Self {
        let matrix_values = matrix.grid.into_iter().flatten().collect();

        Self {
            matrix_values,
        }
    }
}

impl<T> Iterator for MatrixIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.matrix_values.len() == 0 {
            return None;
        }

        Some(self.matrix_values.remove(0))
    }
}

fn are_rows_same_size<T>(arrays: &Vec<Vec<T>>) -> bool {
    if arrays.len() == 1 {
        return true;
    }

    let first_array_len: usize = arrays[0].len();

    for i in 1..arrays.len() {
        if first_array_len != arrays[i].len() {
            return false;
        }
    }

    true
}

fn vec_option_t_to_vec_t<T>(array: Vec<Vec<Option<T>>>) -> Vec<Vec<T>> {
    array
        .into_iter()
        .map(
            |row| row
                .into_iter()
                .map(
                    |cell| cell.unwrap()
                )
                .collect::<Vec<T>>()
        )
        .collect::<Vec<Vec<T>>>()
}

impl<T> Matrix<T>
where
    T: Num + Neg<Output = T> + Clone,
{
    fn determinant(&self) -> Result<T, MatrixError> {
        matrix_determinant(self)
    }
}

fn matrix_determinant<T>(
    matrix: &Matrix<T>,
) -> Result<T, MatrixError>
where
    T: Num + Neg<Output = T> + Clone,
{
    if !matrix.is_squared() {
        return Err(MatrixError::NotSquaredMatrixError);
    }

    if matrix.get_nb_rows() == 1 {
        return Ok(matrix[0][0].clone());
    }

    let mut result: T = T::zero();
    let mut inversion = T::one();

    for (i, value) in matrix.get_row(0).unwrap().into_iter().enumerate() {
        let minor = matrix.minor(0, i).unwrap();
        let minor_determinant =
            matrix_determinant(&minor).unwrap();

        result = result +
            inversion.clone() * value.clone() * minor_determinant;

        inversion = -inversion;
    }

    Ok(result)
}

impl<T> Matrix<T>
where
    T: Clone,
{
    fn minor(
        &self, x: usize, y: usize
    ) -> Result<Self, MatrixError> {
        if self.rows == 1 {
            return Err(MatrixError::MatrixTooSmall);
        }

        if !self.is_squared() {
            return Err(MatrixError::NotSquaredMatrixError);
        }

        let mut new_grid: Vec<Vec<T>> = Vec::new();

        for i in 0..self.rows {
            if x == i {
                continue;
            }

            let mut new_row: Vec<T> = Vec::new();

            for j in 0..self.columns {
                if j == y {
                    continue;
                }

                new_row.push(self.grid[i][j].clone());
            }

            new_grid.push(new_row);
        }

        Ok(Self {
            rows: self.rows - 1,
            columns: self.columns - 1,
            grid: new_grid,
        })
    }

    fn transpose(&self) -> Self {
        let rows = self.columns;
        let columns = self.rows;

        let mut tmp_grid: Vec<Vec<Option<T>>> = vec![vec![None; columns]; rows];

        for i in 0..rows {
            for j in 0..columns {
                tmp_grid[i][j] = Some(self.grid[j][i].clone());
            }
        }

        let grid: Vec<Vec<T>> = vec_option_t_to_vec_t(tmp_grid);

        Self {
            rows,
            columns,
            grid,
        }
    }
}

impl<T> Matrix<T>
where
    T: Num,
{
    fn identity(size: usize) -> Result<Self, MatrixError> {
        if size == 0 {
            return Err(MatrixError::MatrixCantBeZeroSizedError);
        }

        let mut grid: Vec<Vec<T>> = Vec::new();

        for x in 0..size {
            let mut new_row: Vec<T> = Vec::new();

            for y in 0..size {
                new_row.push(
                    if x == y {
                        T::one()
                    }
                    else {
                        T::zero()
                    }
                );
            }

            grid.push(new_row);
        }

        Ok(Self {
            rows: size,
            columns: size,
            grid
        })
    }

    fn is_diagonal(&self) -> bool {
        if !self.is_squared() {
            return false;
        }

        for x in 0..self.rows {
            for y in 0..self.columns {
                if x == y {
                    continue;
                }

                if self.grid[x][y] != T::zero() {
                    return false;
                }
            }
        }

        true
    }
}

impl<T> Matrix<T>
where
    T: Num + Add<Output = T> + Clone,
{
    fn trace(&self) -> Result<T, MatrixError> {
        if !self.is_squared() {
            return Err(MatrixError::NotSquaredMatrixError);
        }

        let mut sum = T::zero();

        for i in 0..self.rows {
            sum = sum + self.grid[i][i].clone();
        }

        Ok(sum)
    }
}

impl<T> Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.rows != other.rows ||
            self.columns != other.columns {
            return Err(MatrixError::NotTheSameSizeError);
        }

        let mut new_grid: Vec<Vec<T>> = Vec::new();

        for x in 0..self.rows {
            let mut new_row: Vec<T> = Vec::new();

            for y in 0..self.columns {
                new_row.push(
                    self.grid[x][y].clone() + other.grid[x][y].clone()
                );
            }

            new_grid.push(new_row);
        }

        Ok(Self {
            rows: self.rows,
            columns: self.columns,
            grid: new_grid,
        })
    }
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + Clone,
{
    fn mul_scalaire(
        &self, scalaire: T,
    ) -> Self {
        let mut new_grid: Vec<Vec<T>> = Vec::new();

        for x in 0..self.rows {
            let mut new_row: Vec<T> = Vec::new();

            for y in 0..self.columns {
                new_row.push(
                    self.grid[x][y].clone() * scalaire.clone()
                );
            }

            new_grid.push(new_row);
        }

        Self {
            rows: self.rows,
            columns: self.columns,
            grid: new_grid,
        }
    }
}

impl<T> Matrix<T>
where
    T: Add<Output = T> + Mul<Output = T> + std::iter::Sum + Clone,
{
    fn mul(
        &self, other: &Self,
    ) -> Result<Self, MatrixError> {
        if self.rows != other.columns ||
            self.columns != other.rows {
            return Err(MatrixError::IncompatibleSizeMulError);
        }

        let rows = self.rows;
        let columns = other.columns;

        let mut new_grid: Vec<Vec<T>> = Vec::new();

        for x in 0..columns {
            let mut new_column: Vec<T> = Vec::new();

            for y in 0..rows {
                let self_row = self.get_row(x).unwrap();
                let other_column = other.get_column(y).unwrap();

                let mul_array = mul_array_elements(self_row, other_column);

                new_column.push(mul_array.into_iter().sum::<T>());
            }

            new_grid.push(new_column);
        }

        Ok(Self {
            rows,
            columns,
            grid: new_grid,
        })
    }
}

fn mul_array_elements<T>(array_1: Vec<&T>, array_2: Vec<&T>) -> Vec<T>
where
    T: Mul<Output = T> + Clone,
{
    let min = std::cmp::min(array_1.len(), array_2.len());
    let mut result: Vec<T> = Vec::new();

    for i in 0..min {
        result.push(array_1[i].clone() * array_2[i].clone());
    }

    result
}

impl<T> Matrix<T>
where
    T: Num + Neg<Output = T> + Clone + Debug,
{
    fn inverse(&self) -> Result<Self, MatrixError> {
        let comatrice = self.comatrice()?;
        let transpose = comatrice.transpose();
        let determinant = self.determinant().unwrap();
        let inverse_determinant = T::one() / determinant;
        Ok(transpose.mul_scalaire(inverse_determinant))
    }
}

impl<T> Matrix<T>
where
    T: Num + Neg<Output = T> + Clone,
{
    fn comatrice(&self) -> Result<Self, MatrixError> {
        if !self.is_squared() {
            return Err(MatrixError::NotSquaredMatrixError);
        }

        if self.rows == 1 {
            return Ok(Matrix::new(vec![vec![T::one()]]).unwrap())
        }

        let mut new_grid: Vec<Vec<T>> = Vec::new();

        for i in 0..self.rows {
            let mut new_row: Vec<T> = Vec::new();

            for j in 0..self.columns {
                let sign: T = get_sign_from_pos(i+j);
                let cell = sign * self.minor(i, j)
                    .unwrap()
                    .determinant()
                    .unwrap();
                new_row.push(cell);
            }

            new_grid.push(new_row);
        }

        Ok(Matrix::new(new_grid).unwrap())
    }
}

fn get_sign_from_pos<T>(pos: usize) -> T
where
    T: Num + Neg<Output = T>,
{
    if pos % 2 == 1 {
        -T::one()
    }
    else {
        T::one()
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = Vec<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.grid[index]
    }
}

impl<T> Matrix<T>
where
    T: Debug,
{
    fn print(&self) {
        for i in 0..self.rows {
            println!("{:?}", self.grid[i]);
        }
    }
}

#[cfg(test)]
mod test_matrix {
    use super::*;

    #[test]
    fn test_are_arrays_same_size() {
        let one_row = vec![vec![1, 2, 3]];
        let same_rows_size = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let different_rows_size = vec![vec![1, 2], vec![3, 4, 5]];

        assert!(are_rows_same_size(&one_row));
        assert!(are_rows_same_size(&same_rows_size));
        assert!(!are_rows_same_size(&different_rows_size));
    }

    #[test]
    fn test_new() {
        let empty: Vec<Vec<usize>> = Vec::new();
        let different_rows_size = vec![vec![1, 2], vec![3, 4, 5]];
        let good = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

        assert_eq!(Matrix::new(empty), Err(MatrixError::EmptyMatrixError));
        assert_eq!(
            Matrix::new(different_rows_size),
            Err(MatrixError::RowsNotOfSameSizeError)
        );
        assert!(Matrix::new(good).is_ok());
    }

    #[test]
    fn test_from_vec() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let a_res = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let matrix_a = Matrix::new(a_res).unwrap();

        let b = vec![1, 2, 3, 4, 5, 6, 7, 8];

        assert_eq!(Matrix::from_vec(a, 3).unwrap(), matrix_a);
        assert_eq!(Matrix::from_vec(b, 3), Err(MatrixError::InvalidVecSize));
    }

    #[test]
    fn test_add() {
        let first = vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]];
        let wrong_size = vec![vec![2, 2], vec![2, 2], vec![2, 2]];
        let second = vec![vec![2, 2, 2], vec![2, 2, 2], vec![2, 2, 2]];
        let result = vec![vec![3, 3, 3], vec![3, 3, 3], vec![3, 3, 3]];

        let matrix_first = Matrix::new(first).unwrap();
        let matrix_wrong_size = Matrix::new(wrong_size).unwrap();
        let matrix_second = Matrix::new(second).unwrap();
        let matrix_result = Matrix::new(result).unwrap();

        assert_eq!(
            matrix_first.add(&matrix_wrong_size),
            Err(MatrixError::NotTheSameSizeError)
        );

        assert_eq!(
            matrix_first.add(&matrix_second),
            Ok(matrix_result)
        );
    }

    #[test]
    fn test_mul_scalaire() {
        let scalaire = 2;
        let grid = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let matrix = Matrix::new(grid).unwrap();
        let result = vec![vec![2, 4, 6], vec![8, 10, 12], vec![14, 16, 18]];
        let matrix_result = Matrix::new(result).unwrap();

        assert_eq!(
            matrix.mul_scalaire(scalaire),
            matrix_result
        );
    }

    #[test]
    fn test_mul() {
        let a = Matrix::new(vec![vec![1, 2, 0], vec![4, 3, -1]]).unwrap();
        let b =
            Matrix::new(vec![vec![5, 1], vec![2, 3], vec![3, 4]]).unwrap();
        let a_mul_b =
            Matrix::new(vec![vec![9, 7], vec![23, 9]]).unwrap();
        let b_mul_a = Matrix::new(
                vec![vec![9, 13, -1], vec![14, 13, -3], vec![19, 18, -4]]
            ).unwrap();

        assert_eq!(a.mul(&b).unwrap(), a_mul_b);
        assert_eq!(b.mul(&a).unwrap(), b_mul_a);
        assert_eq!(a.mul(&a), Err(MatrixError::IncompatibleSizeMulError));
    }

    #[test]
    fn test_index() {
        let matrix =
            Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
            .unwrap();

        assert_eq!(matrix[0], vec![1, 2, 3]);
        assert_eq!(matrix[1], vec![4, 5, 6]);
        assert_eq!(matrix[2], vec![7, 8, 9]);
    }

    #[test]
    fn test_get_column() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let matrix = Matrix::new(matrix).unwrap();

        assert_eq!(matrix.get_column(1).unwrap(), vec![&2, &5, &8]);
        assert_eq!(matrix.get_column(3), None);
    }

    #[test]
    fn test_identity() {
        let result = Matrix::new(
            vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
        ).unwrap();

        assert_eq!(Matrix::identity(3).unwrap(), result);
        assert_eq!(
            Matrix::<usize>::identity(0),
            Err(MatrixError::MatrixCantBeZeroSizedError)
        );
    }

    #[test]
    fn test_is_diagonal() {
        let non_square =
            Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .unwrap();
        let square_not_diagonal =
            Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
            .unwrap();
        let diagonal =
            Matrix::new(vec![vec![1, 0, 0], vec![0, 5, 0], vec![0, 0, 9]])
            .unwrap();

        assert!(!non_square.is_diagonal());
        assert!(!square_not_diagonal.is_diagonal());
        assert!(diagonal.is_diagonal());
    }

    #[test]
    fn test_minor() {
        let matrix =
            Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
            .unwrap();
        let zero_zero =
            Matrix::new(vec![vec![5, 6], vec![8, 9]])
            .unwrap();
        let zero_one =
            Matrix::new(vec![vec![4, 6], vec![7, 9]])
            .unwrap();
        let one_two =
            Matrix::new(vec![vec![1, 2], vec![7, 8]])
            .unwrap();

        let too_small =
            Matrix::new(vec![vec![1]])
            .unwrap();

        assert_eq!(matrix.minor(0, 0), Ok(zero_zero));
        assert_eq!(matrix.minor(0, 1), Ok(zero_one));
        assert_eq!(matrix.minor(1, 2), Ok(one_two));

        assert_eq!(
            too_small.minor(0, 0),
            Err(MatrixError::MatrixTooSmall)
        );
    }

    #[test]
    fn test_determinant() {
        let moins_dix_sept = Matrix::new(
            vec![vec![1, 3], vec![7, 4]]
        ).unwrap();

        let dix_huit = Matrix::new(
            vec![vec![-2, 2, -3], vec![-1, 1, 3], vec![2, 0, -1]]
        ).unwrap();

        let identity = Matrix::identity(4).unwrap();

        let moins_trois_cent_six = Matrix::new(
            vec![vec![6, 1, 1], vec![4, -2, 5], vec![2, 8, 7]]
        ).unwrap();

        let cent_huit = Matrix::new(
            vec![
                vec![2, 4, 5, 6],
                vec![-1, 5, 6, 9],
                vec![3, 7, 1, -6],
                vec![4, -2, 3, 5],
            ]
        ).unwrap();

        let un = Matrix::new(
            vec![vec![1]]
        ).unwrap();

        let not_squared = Matrix::new(
            vec![vec![-2, 2, -3], vec![4, 0, -1]]
        ).unwrap();

        assert_eq!(moins_dix_sept.determinant(), Ok(-17));
        assert_eq!(dix_huit.determinant(), Ok(18));
        assert_eq!(identity.determinant(), Ok(1));
        assert_eq!(moins_trois_cent_six.determinant(), Ok(-306));
        assert_eq!(cent_huit.determinant(), Ok(108));
        assert_eq!(un.determinant(), Ok(1));
        assert_eq!(
            not_squared.determinant(),
            Err(MatrixError::NotSquaredMatrixError)
        );
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        ).unwrap();
        let a_result = Matrix::new(
            vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]]
        ).unwrap();

        let b = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]]
        ).unwrap();
        let b_result = Matrix::new(
            vec![vec![1, 4], vec![2, 5], vec![3, 6]]
        ).unwrap();
        let c = Matrix::new(
            vec![vec![1]]
        ).unwrap();

        assert_eq!(a.transpose(), a_result);
        assert_eq!(b.transpose(), b_result);
        assert_eq!(c.transpose(), c);
    }

    #[test]
    fn test_comatrice() {
        let a = Matrix::new(
            vec![vec![1, 2], vec![3, 4]]
        ).unwrap();
        let a_result = Matrix::new(
            vec![vec![4, -3], vec![-2, 1]]
        ).unwrap();

        let b = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        ).unwrap();
        let b_result = Matrix::new(
            vec![vec![-3, 6, -3], vec![6, -12, 6], vec![-3, 6, -3]]
        ).unwrap();

        let c = Matrix::new(
            vec![
                vec![1, 2, 341, 4],
                vec![8, 5, 2, 0],
                vec![9, 5, 3, 2],
                vec![7, 5, 6, 5]
            ]
        ).unwrap();
        let c_result = Matrix::new(
            vec![
                vec![-15, 38, -35, 25],
                vec![5049, -10436, 7, 3359],
                vec![-8425, 13494, -35, -1657],
                vec![3382, -5428, 42, -1712]
            ]
        ).unwrap();

        let one_value = Matrix::new(vec![vec![42]]).unwrap();
        let one_value_result = Matrix::new(vec![vec![1]]).unwrap();

        let not_squared = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]]
        ).unwrap();

        assert_eq!(a.comatrice(), Ok(a_result));
        assert_eq!(b.comatrice(), Ok(b_result));
        assert_eq!(c.comatrice(), Ok(c_result));
        assert_eq!(one_value.comatrice(), Ok(one_value_result));
        assert_eq!(
            not_squared.comatrice(),
            Err(MatrixError::NotSquaredMatrixError)
        );
    }

    #[test]
    fn test_get_sign_from_pos() {
        assert_eq!(get_sign_from_pos::<isize>(1), -1_isize);
        assert_eq!(get_sign_from_pos::<isize>(2), 1_isize);
    }

    #[test]
    fn test_inverse() {
        let a = Matrix::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        ).unwrap();
        let a_result = Matrix::new(
            vec![vec![-2.0, 1.0], vec![3.0/2.0, -1.0/2.0]]
        ).unwrap();

        let b = Matrix::new(
            vec![vec![1, 2, 3], vec![0, 1, 4], vec![5, 6, 0]]
        ).unwrap();
        let b_result = Matrix::new(
            vec![vec![-24, 18, 5], vec![20, -15, -4], vec![-5, 4, 1]]
        ).unwrap();

        let not_squared = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]]
        ).unwrap();

        assert_eq!(a.inverse(), Ok(a_result));
        assert_eq!(b.inverse(), Ok(b_result));
        assert_eq!(
            not_squared.inverse(),
            Err(MatrixError::NotSquaredMatrixError)
        );
    }

    #[test]
    fn test_trace() {
        let not_squared = Matrix::new(
            vec![vec![1, 4], vec![2, 5], vec![3, 6]]
        ).unwrap();
        let a = Matrix::new(
            vec![vec![1, 2], vec![3, 4]]
        ).unwrap();
        let b = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        ).unwrap();

        assert_eq!(
            not_squared.trace(),
            Err(MatrixError::NotSquaredMatrixError)
        );
        assert_eq!(a.trace(), Ok(5));
        assert_eq!(b.trace(), Ok(15))
    }
}













