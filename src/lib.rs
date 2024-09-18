use std::ops::{
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Index,
};
use std::error::Error;
use std::fmt::{Display, Debug, Formatter, Error as FmtError};
use math_concept::{
    zero::Zero,
    one::One,
};

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix<T> {
    rows: usize,
    columns: usize,
    grid: Vec<Vec<T>>,
}

#[derive(Debug, PartialEq)]
pub enum MatrixError {
    EmptyMatrixError,
    RowsNotOfSameSizeError,
    NotTheSameSizeError,
    IncompatibleSizeMulError,
    MatrixCantBeZeroSizedError,
    NotSquaredMatrixError,
    MatrixTooSmallError,
    InvalidVecSizeError,
    NoDecompositionError,
    DeterminantIsZeroError,
}

impl MatrixError {
    const EMPTY_MATRIX_ERROR_STR: &'static str = "Matrix is empty.";
    const ROWS_NOT_OF_SAME_SIZE_ERROR_STR: &'static str =
        "The rows are not of the same size";
    const NOT_THE_SAME_SIZE_ERROR_STR: &'static str =
        "The Matrixes have different size.";
    const INCOMPATIBLE_SIZE_MUL_ERROR_STR: &'static str =
        "The Matrixes are not compatible.";
    const MATRIX_CANT_BE_ZERO_SIDED_ERROR_STR: &'static str =
        "Matrix can't be of zero size.";
    const NOT_SQUARED_MATRIX_ERROR_STR: &'static str = "Matrix is not squared.";
    const MATRIX_TOO_SMALL_ERROR_STR: &'static str = "Matrix is too small.";
    const INVALID_VEC_SIZE_ERROR_STR: &'static str = "Vec of invalid size.";
    const NO_DECOMPOSITION_ERROR_STR: &'static str = "No decomposition lu.";
    const DETERMINANT_IS_ZERO_ERROR_STR: &'static str = "Determinant is zero.";
}

impl Display for MatrixError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::EmptyMatrixError =>
                write!(f, "{}", MatrixError::EMPTY_MATRIX_ERROR_STR),
            Self::RowsNotOfSameSizeError =>
                write!(f, "{}", MatrixError::ROWS_NOT_OF_SAME_SIZE_ERROR_STR),
            Self::NotTheSameSizeError =>
                write!(f, "{}", MatrixError::NOT_THE_SAME_SIZE_ERROR_STR),
            Self::IncompatibleSizeMulError =>
                write!(f, "{}", MatrixError::INCOMPATIBLE_SIZE_MUL_ERROR_STR),
            Self::MatrixCantBeZeroSizedError =>
                write!(f, "{}",
                    MatrixError::MATRIX_CANT_BE_ZERO_SIDED_ERROR_STR),
            Self::NotSquaredMatrixError =>
                write!(f, "{}", MatrixError::NOT_SQUARED_MATRIX_ERROR_STR),
            Self::MatrixTooSmallError =>
                write!(f, "{}", MatrixError::MATRIX_TOO_SMALL_ERROR_STR),
            Self::InvalidVecSizeError =>
                write!(f, "{}", MatrixError::INVALID_VEC_SIZE_ERROR_STR),
            Self::NoDecompositionError =>
                write!(f, "{}", MatrixError::NO_DECOMPOSITION_ERROR_STR),
            Self::DeterminantIsZeroError =>
                write!(f, "{}", MatrixError::DETERMINANT_IS_ZERO_ERROR_STR),
        }
    }
}

impl Error for MatrixError {}

impl<T> Matrix<T> {
    pub fn new(grid: Vec<Vec<T>>) -> Result<Self, MatrixError> {
        if grid.is_empty() {
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

    pub fn from_vec(
        v: Vec<T>, rows: usize
    ) -> Result<Self, MatrixError> {
        let len: f32 = v.len() as f32;
        let x_f32: f32 = rows as f32;
        if (len / x_f32) % 1. != 0. {
            return Err(MatrixError::InvalidVecSizeError);
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

    pub fn get_nb_rows(&self) -> usize {
        self.rows
    }

    pub fn get_nb_columns(&self) -> usize {
        self.columns
    }

    pub fn get_row(&self, index: usize) -> Option<Vec<&T>> {
        if index >= self.rows {
            None
        }
        else {
            Some(self.grid[index].iter().collect::<Vec<&T>>())
        }
    }

    pub fn get_column(&self, index: usize) -> Option<Vec<&T>> {
        if index >= self.columns {
            return None;
        }

        let mut column: Vec<&T> = Vec::new();

        for x in 0..self.rows {
            column.push(&self.grid[x][index]);
        }

        Some(column)
    }

    pub fn get_diagonal(&self) -> Vec<&T> {
        let min = std::cmp::min(self.rows, self.columns);
        let mut diagonal: Vec<&T> = Vec::new();

        for i in 0..min {
            diagonal.push(&self.grid[i][i]);
        }

        diagonal
    }

    pub fn is_squared(&self) -> bool {
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

pub struct MatrixIterator<T> {
    matrix_values: Vec<T>,
}

impl<T> MatrixIterator<T> {
    pub fn new(matrix: Matrix<T>) -> Self {
        let matrix_values = matrix.grid.into_iter().flatten().collect();

        Self {
            matrix_values,
        }
    }
}

impl<T> Iterator for MatrixIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.matrix_values.is_empty() {
            return None;
        }

        Some(self.matrix_values.remove(0))
    }
}

fn are_rows_same_size<T>(arrays: &[Vec<T>]) -> bool {
    if arrays.len() == 1 {
        return true;
    }

    let first_array_len: usize = arrays[0].len();

    for row in arrays.iter().skip(1) {
        if first_array_len != row.len() {
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
    T: Zero + One + Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Clone,
{
    pub fn determinant(&self) -> Result<T, MatrixError> {
        matrix_determinant(self)
    }

    pub fn is_invertible(&self) -> bool {
        if self.rows != self.columns {
            false
        }
        else {
            !self.determinant().unwrap().is_zero()
        }
    }
}

fn matrix_determinant<T>(
    matrix: &Matrix<T>,
) -> Result<T, MatrixError>
where
    T: Zero + One + Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Clone,
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
    pub fn minor(
        &self, x: usize, y: usize
    ) -> Result<Self, MatrixError> {
        if self.rows == 1 {
            return Err(MatrixError::MatrixTooSmallError);
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

    pub fn transpose(&self) -> Self {
        let rows = self.columns;
        let columns = self.rows;

        let mut tmp_grid: Vec<Vec<Option<T>>> = vec![vec![None; columns]; rows];

        for (i, row) in tmp_grid.iter_mut().enumerate().take(rows) {
            for (j, cell) in row.iter_mut().enumerate().take(columns) {
                *cell = Some(self.grid[j][i].clone());
            }
        }

        let grid: Vec<Vec<T>> = vec_option_t_to_vec_t(tmp_grid);

        Self {
            rows,
            columns,
            grid,
        }
    }

    pub fn as_vec(&self) -> Vec<Vec<T>> {
        self.grid.clone()
    }
}

impl<T> Matrix<T>
where
    T: Zero + One,
{
    pub fn identity(size: usize) -> Result<Self, MatrixError> {
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
}

impl<T> Matrix<T>
where
    T: Zero + PartialEq,
{
    pub fn is_diagonal(&self) -> bool {
        if !self.is_squared() {
            return false;
        }

        for x in 0..self.rows {
            for y in 0..self.columns {
                if x == y {
                    continue;
                }

                if !self.grid[x][y].is_zero() {
                    return false;
                }
            }
        }

        true
    }
}

impl<T> Matrix<T>
where
    T: Zero + Add<Output = T> + Clone,
{
    pub fn trace(&self) -> Result<T, MatrixError> {
        if !self.is_squared() {
            return Err(MatrixError::NotSquaredMatrixError);
        }

        Ok(
            sum_vec(
                &self.get_diagonal()
                    .into_iter()
                    .cloned()
                    .collect()
            )
        )
    }
}

fn sum_vec<T>(vec: &Vec<T>) -> T
where
    T: Zero + Add<Output = T> + Clone,
{
    let mut sum = T::zero();

    for val in vec {
        sum = sum + val.clone().clone();
    }

    sum
}

impl<T> Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    pub fn add_to(&self, other: &Self) -> Result<Self, MatrixError> {
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

impl<T> Add for Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.add_to(&other).unwrap()
    }
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + Clone,
{
    pub fn scale(
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
    T: Zero + Add<Output = T> + Mul<Output = T> + Clone,
{
    pub fn mul_to(
        &self, other: &Self,
    ) -> Result<Self, MatrixError> {
        if self.columns != other.rows {
            return Err(MatrixError::IncompatibleSizeMulError);
        }

        let rows = self.rows;
        let columns = other.columns;

        let mut new_grid: Vec<Vec<T>> = Vec::new();

        for x in 0..rows {
            let mut new_column: Vec<T> = Vec::new();

            for y in 0..columns {
                let self_row = self.get_row(x).unwrap();
                let other_column = other.get_column(y).unwrap();

                let mul_array = mul_array_elements(self_row, other_column);

                new_column.push(sum_vec(&mul_array));
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

impl<T> Mul for Matrix<T>
where
    T: Zero + Add<Output = T> + Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.mul_to(&other).unwrap()
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
    T: Zero + One +
        Add<Output = T> + Mul<Output = T> + Div<Output = T> +
        Neg<Output = T> + Clone,
{
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if !self.is_squared() {
            return Err(MatrixError::NotSquaredMatrixError);
        }
        let determinant = self.determinant()?;
        if determinant.is_zero() {
            return Err(MatrixError::DeterminantIsZeroError);
        }
        let comatrice = self.comatrice()?;
        let transpose = comatrice.transpose();
        let inverse_determinant = T::one() / determinant;
        Ok(transpose.scale(inverse_determinant))
    }
}

impl<T> Matrix<T>
where
    T: Zero + One + Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Clone,
{
    pub fn comatrice(&self) -> Result<Self, MatrixError> {
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
    T: One + Neg<Output = T>,
{
    if pos % 2 == 1 {
        -T::one()
    }
    else {
        T::one()
    }
}

impl<T> Sub for Matrix<T>
where
    T: One + Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Clone,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + other.scale(-T::one())
    }
}

impl<T> Matrix<T>
where
    T: Zero + One +
        Add<Output = T> + Mul<Output = T> + Div<Output = T> +
        Neg<Output = T> + Clone,
{
    pub fn div_to(&self, other: &Self) -> Result<Self, MatrixError> {
        self.mul_to(&other.inverse()?)
    }
}

impl<T> Div for Matrix<T>
where
    T: Zero + One +
    Add<Output = T> + Mul<Output = T> + Div<Output = T> +
    Neg<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.div_to(&other).unwrap()
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
    pub fn print(&self) {
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
        assert_eq!(
            Matrix::from_vec(b, 3), Err(MatrixError::InvalidVecSizeError)
        );
    }

    #[test]
    fn test_add_to() {
        let first = vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]];
        let wrong_size = vec![vec![2, 2], vec![2, 2], vec![2, 2]];
        let second = vec![vec![2, 2, 2], vec![2, 2, 2], vec![2, 2, 2]];
        let result = vec![vec![3, 3, 3], vec![3, 3, 3], vec![3, 3, 3]];

        let matrix_first = Matrix::new(first).unwrap();
        let matrix_wrong_size = Matrix::new(wrong_size).unwrap();
        let matrix_second = Matrix::new(second).unwrap();
        let matrix_result = Matrix::new(result).unwrap();

        assert_eq!(
            matrix_first.add_to(&matrix_wrong_size),
            Err(MatrixError::NotTheSameSizeError)
        );

        assert_eq!(
            matrix_first.add_to(&matrix_second),
            Ok(matrix_result)
        );
    }

    #[test]
    #[should_panic]
    fn test_add() {
        let first = vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]];
        let wrong_size = vec![vec![2, 2], vec![2, 2], vec![2, 2]];
        let second = vec![vec![2, 2, 2], vec![2, 2, 2], vec![2, 2, 2]];
        let result = vec![vec![3, 3, 3], vec![3, 3, 3], vec![3, 3, 3]];

        let matrix_first = Matrix::new(first).unwrap();
        let matrix_wrong_size = Matrix::new(wrong_size).unwrap();
        let matrix_second = Matrix::new(second).unwrap();
        let matrix_result = Matrix::new(result).unwrap();

        assert_eq!(matrix_first.clone() + matrix_second, matrix_result);
        let _ = matrix_first + matrix_wrong_size;
    }

    #[test]
    fn test_scale() {
        let scalaire = 2;
        let grid = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let matrix = Matrix::new(grid).unwrap();
        let result = vec![vec![2, 4, 6], vec![8, 10, 12], vec![14, 16, 18]];
        let matrix_result = Matrix::new(result).unwrap();

        assert_eq!(
            matrix.scale(scalaire),
            matrix_result
        );
    }

    #[test]
    fn test_mul_to() {
        let a = Matrix::new(vec![vec![1, 2, 0], vec![4, 3, -1]]).unwrap();
        let b =
            Matrix::new(vec![vec![5, 1], vec![2, 3], vec![3, 4]]).unwrap();
        let a_mul_b =
            Matrix::new(vec![vec![9, 7], vec![23, 9]]).unwrap();
        let b_mul_a = Matrix::new(
                vec![vec![9, 13, -1], vec![14, 13, -3], vec![19, 18, -4]]
            ).unwrap();

        assert_eq!(a.mul_to(&b).unwrap(), a_mul_b);
        assert_eq!(b.mul_to(&a).unwrap(), b_mul_a);
        assert_eq!(a.mul_to(&a), Err(MatrixError::IncompatibleSizeMulError));
    }

    #[test]
    #[should_panic]
    fn test_mul() {
        let a = Matrix::new(vec![vec![1, 2, 0], vec![4, 3, -1]]).unwrap();
        let b =
        Matrix::new(vec![vec![5, 1], vec![2, 3], vec![3, 4]]).unwrap();
        let a_mul_b =
        Matrix::new(vec![vec![9, 7], vec![23, 9]]).unwrap();
        let b_mul_a = Matrix::new(
            vec![vec![9, 13, -1], vec![14, 13, -3], vec![19, 18, -4]]
        ).unwrap();

        assert_eq!(a.clone() * b.clone(), a_mul_b);
        assert_eq!(b * a.clone(), b_mul_a);
        let _ = a.clone() * a;
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
    fn test_get_diagonal() {
        let a = Matrix::new(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let b = Matrix::new(vec![vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
        let c = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        ).unwrap();

        assert_eq!(a.get_diagonal(), vec![&1, &5]);
        assert_eq!(b.get_diagonal(), vec![&1, &4]);
        assert_eq!(c.get_diagonal(), vec![&1, &5, &9]);
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
        let non_square = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]]
        ).unwrap();
        let square_not_diagonal = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        ).unwrap();
        let diagonal_1 = Matrix::new(
            vec![vec![1, 0, 0], vec![0, 5, 0], vec![0, 0, 9]]
        ).unwrap();
        let diagonal_2 = Matrix::new(
            vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
        ).unwrap();

        assert!(!non_square.is_diagonal());
        assert!(!square_not_diagonal.is_diagonal());
        assert!(diagonal_1.is_diagonal());
        assert!(diagonal_2.is_diagonal());
    }

    #[test]
    fn test_is_invertible() {
        let oui = Matrix::new(
            vec![vec![1, 3], vec![7, 4]]
        ).unwrap();

        let singuliere = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        ).unwrap();

        let mauvaise_dimensions = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]]
        ).unwrap();

        assert!(oui.is_invertible());
        assert!(!singuliere.is_invertible());
        assert!(!mauvaise_dimensions.is_invertible());
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
            Err(MatrixError::MatrixTooSmallError)
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

        let c = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        ).unwrap();

        let not_squared = Matrix::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]]
        ).unwrap();

        assert_eq!(a.inverse(), Ok(a_result));
        assert_eq!(b.inverse(), Ok(b_result));
        assert_eq!(c.inverse(), Err(MatrixError::DeterminantIsZeroError));
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

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: NotTheSameSizeError")]
    fn test_sub() {
        let a = Matrix::new(
            vec![
                vec![ 0,  4],
                vec![-3, -3],
            ]
        ).unwrap();
        let b = Matrix::new(
            vec![
                vec![-9,  0],
                vec![ 2, -2],
            ]
        ).unwrap();
        let a_minux_b = Matrix::new(
            vec![
                vec![ 9,  4],
                vec![-5, -1],
            ]
        ).unwrap();
        let c = Matrix::new(
            vec![
                vec![ 0,  4],
                vec![-3, -3],
                vec![ 1,  2],
            ]
        ).unwrap();

        assert_eq!(a.clone() - b, a_minux_b);
        let _ = a - c;
    }

    #[test]
    fn test_div_to() {
        let a = Matrix::new(
            vec![
                vec![3., -2.],
                vec![4., -3.],
            ]
        ).unwrap();
        let b = Matrix::new(
            vec![
                vec![6., -10.],
                vec![1., -2.],
            ]
        ).unwrap();
        let a_div_b = Matrix::new(
            vec![
                vec![2., -9.],
                vec![2.5, -11.],
            ]
        ).unwrap();

        let zero_deter = Matrix::new(
            vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]]
        ).unwrap();

        let not_squared = Matrix::new(
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        ).unwrap();

        let moins_trois_cent_six = Matrix::new(
            vec![vec![6., 1., 1.], vec![4., -2., 5.], vec![2., 8., 7.]]
        ).unwrap();

        assert_eq!(
            a.clone().div_to(&zero_deter),
            Err(MatrixError::DeterminantIsZeroError)
        );
        assert_eq!(
            a.clone().div_to(&not_squared),
            Err(MatrixError::NotSquaredMatrixError)
        );
        assert_eq!(
            b.clone().div_to(&moins_trois_cent_six),
            Err(MatrixError::IncompatibleSizeMulError)
        );
        assert_eq!(a.div_to(&b), Ok(a_div_b));
    }

    #[test]
    fn test_div() {
        let a = Matrix::new(
            vec![
                vec![3., -2.],
                vec![4., -3.],
            ]
        ).unwrap();
        let b = Matrix::new(
            vec![
                vec![6., -10.],
                vec![1., -2.],
            ]
        ).unwrap();

        let a_div_b = Matrix::new(
            vec![
                vec![2., -9.],
                vec![2.5, -11.],
            ]
        ).unwrap();

        assert_eq!(a / b, a_div_b);
    }

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: DeterminantIsZeroError")]
    fn test_div_zero_deter() {
        let a = Matrix::new(
            vec![
                vec![3., -2.],
                vec![4., -3.],
            ]
        ).unwrap();

        let zero_deter = Matrix::new(
            vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]]
        ).unwrap();

        let _ = a / zero_deter;
    }

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: NotSquaredMatrixError")]
    fn test_div_not_squared() {
        let a = Matrix::new(
            vec![
                vec![3., -2.],
                vec![4., -3.],
            ]
        ).unwrap();
        let not_squared = Matrix::new(
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        ).unwrap();

        let _ = a / not_squared;
    }

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: IncompatibleSizeMulError")]
    fn test_div_imcompatible_size() {
        let a = Matrix::new(
            vec![
                vec![3., -2.],
                vec![4., -3.],
            ]
        ).unwrap();

        let wrong_size = Matrix::new(
            vec![vec![6., 1., 1.], vec![4., -2., 5.], vec![2., 8., 7.]]
        ).unwrap();

        let _ = a / wrong_size;
    }
}













