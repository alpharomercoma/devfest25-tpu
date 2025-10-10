// A simple processor in our array.
class ProcessingElement {
  constructor(name) {
    this.name = name;
    this.accumulator = 0;
  }

  // Each "tick", the PE multiplies inputs and adds to its sum.
  compute(valA, valB) {
    if (valA === undefined || valB === undefined) return; // Skip if no real data

    console.log(`  > ${this.name} computes: ${this.accumulator} + (${valA} * ${valB})`);
    this.accumulator += valA * valB;
  }
}

// ---- Data and PE Setup ----
const A = [[1, 2], [3, 4]];
const B = [[5, 6], [7, 8]];

// Create our 2x2 grid of Processing Elements (PEs)
const PEs = [
  [new ProcessingElement('PE(0,0)'), new ProcessingElement('PE(0,1)')],
  [new ProcessingElement('PE(1,0)'), new ProcessingElement('PE(1,1)')]
];

// ---- Simulation ----
console.log('--- Systolic Array Simulation Start ---');

const total_ticks = 5; // Need 5 ticks for 2x2 matrix multiplication

for (let tick = 0; tick < total_ticks; tick++) {
  console.log(`\n--- Clock Tick ${tick + 1} ---`);

  // Process each PE
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      // Calculate which element of A and B should arrive at PE(i,j) at this tick
      // PE(i,j) starts receiving data at tick (i+j)
      // k represents which element in the dot product we're computing
      const k = tick - (i + j);

      if (k >= 0 && k < 2) { // Valid range for our 2x2 matrices
        const a_val = A[i][k];
        const b_val = B[k][j];

        PEs[i][j].compute(a_val, b_val);
      }
    }
  }

  // Print the state of all accumulators at the end of the tick
  const currentState = PEs.map(row => row.map(pe => pe.accumulator));
  console.log('  Accumulator State:', JSON.stringify(currentState));
}

// ---- Final Result ----
const C = PEs.map(row => row.map(pe => pe.accumulator));

console.log('\n--- Simulation Complete ---');
console.log('Final Result Matrix C:');
console.log(JSON.stringify(C));
console.log('\nExpected (A x B):');
console.log('[[19, 22], [43, 50]]');