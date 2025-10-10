const t1 = [[1, 2]]; // Shape: 1x2
const t2 = [[3], [4]]; // Shape: 2x1

// Multiply corresponding elements and sum them up
const innerProduct_scalar = (t1[0][0] * t2[0][0]) + (t1[0][1] * t2[1][0]);
// (1 * 3) + (2 * 4) = 3 + 8 = 11

// The result is a 1x1 matrix
const innerProduct_matrix = [[innerProduct_scalar]];

console.log(innerProduct_matrix); // Outputs: [[11]]