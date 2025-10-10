const t1 = [[1, 2]]; // Shape: 1x2
const t2 = [[3], [4]]; // Shape: 2x1

// The outer product of t2 and t1
const outerProduct = [
  // First row of t2 * the row vector t1
  [(t2[0][0] * t1[0][0]), (t2[0][0] * t1[0][1])], // (3 * 1), (3 * 2)

  // Second row of t2 * the row vector t1
  [(t2[1][0] * t1[0][0]), (t2[1][0] * t1[0][1])]  // (4 * 1), (4 * 2)
];

console.log(outerProduct); // Outputs: [[3, 6], [4, 8]]