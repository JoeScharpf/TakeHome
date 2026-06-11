/**
 * Vicsek model engine — mirrors vicsek.py step() and polarization().
 * Problem 1 browser demo only; qualitative match to Python, not bit-identical RNG.
 */

function mulberry32(seed) {
  let t = seed >>> 0;
  return function next() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function wrapMod(x, L) {
  return ((x % L) + L) % L;
}

function minimumImageDelta(a, b, L) {
  let dx = a - b;
  dx -= L * Math.round(dx / L);
  return dx;
}

class VicsekSim {
  constructor({ N = 125, R = 1.0, v = 1.0, eta = 0.5, L = 5.0, dt = 0.25, seed = 42 } = {}) {
    this.L = L;
    this.dt = dt;
    this.N = N;
    this.R = R;
    this.v = v;
    this.eta = eta;
    this.seed = seed;
    this.stepCount = 0;
    this.x = new Float64Array(N);
    this.y = new Float64Array(N);
    this.theta = new Float64Array(N);
    this._rng = mulberry32(seed);
    this.reset(seed);
  }

  reset(seed = this.seed) {
    this.seed = seed;
    this._rng = mulberry32(seed);
    this.stepCount = 0;
    const L = this.L;
    for (let i = 0; i < this.N; i += 1) {
      this.x[i] = this._rng() * L;
      this.y[i] = this._rng() * L;
      this.theta[i] = this._rng() * 2 * Math.PI - Math.PI;
    }
  }

  resize(N) {
    this.N = N;
    this.x = new Float64Array(N);
    this.y = new Float64Array(N);
    this.theta = new Float64Array(N);
    this.reset(this.seed);
  }

  polarization() {
    let sumCos = 0;
    let sumSin = 0;
    const n = this.N;
    for (let i = 0; i < n; i += 1) {
      sumCos += Math.cos(this.theta[i]);
      sumSin += Math.sin(this.theta[i]);
    }
    sumCos /= n;
    sumSin /= n;
    return Math.hypot(sumCos, sumSin);
  }

  step() {
    const { N, L, R, v, eta, dt } = this;
    const R2 = R * R;
    const newTheta = new Float64Array(N);
    const newX = new Float64Array(N);
    const newY = new Float64Array(N);

    for (let i = 0; i < N; i += 1) {
      let sumCos = 0;
      let sumSin = 0;
      let count = 0;
      const xi = this.x[i];
      const yi = this.y[i];

      for (let j = 0; j < N; j += 1) {
        const dx = minimumImageDelta(xi, this.x[j], L);
        const dy = minimumImageDelta(yi, this.y[j], L);
        if (dx * dx + dy * dy < R2) {
          sumCos += Math.cos(this.theta[j]);
          sumSin += Math.sin(this.theta[j]);
          count += 1;
        }
      }

      const alignX = (v * sumCos) / count;
      const alignY = (v * sumSin) / count;
      const xiNoise = this._rng() * 2 * Math.PI - Math.PI;
      const noiseX = eta * Math.cos(xiNoise);
      const noiseY = eta * Math.sin(xiNoise);
      newTheta[i] = Math.atan2(alignY + noiseY, alignX + noiseX);

      const oldTheta = this.theta[i];
      newX[i] = wrapMod(xi + v * dt * Math.cos(oldTheta), L);
      newY[i] = wrapMod(yi + v * dt * Math.sin(oldTheta), L);
    }

    this.x.set(newX);
    this.y.set(newY);
    this.theta.set(newTheta);
    this.stepCount += 1;
  }
}
