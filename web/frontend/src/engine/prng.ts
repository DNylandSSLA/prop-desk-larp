/**
 * Seeded PRNG using mulberry32 + Box-Muller for normal distribution.
 * Deterministic per-seed so each browser refresh produces a new but
 * reproducible simulation (seeded from Date.now()).
 */

export class PRNG {
  private state: number;

  constructor(seed: number) {
    this.state = seed | 0;
  }

  /** Uniform [0, 1) */
  random(): number {
    this.state |= 0;
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Normal(mean, std) via Box-Muller */
  normal(mean = 0, std = 1): number {
    let u1 = this.random();
    let u2 = this.random();
    // Avoid log(0)
    while (u1 === 0) u1 = this.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + z * std;
  }

  /** Uniform integer in [lo, hi) */
  randInt(lo: number, hi: number): number {
    return lo + Math.floor(this.random() * (hi - lo));
  }

  /** Pick n unique items from array (Fisher-Yates partial shuffle) */
  sample<T>(arr: readonly T[], n: number): T[] {
    const copy = arr.slice();
    const count = Math.min(n, copy.length);
    for (let i = 0; i < count; i++) {
      const j = i + Math.floor(this.random() * (copy.length - i));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy.slice(0, count);
  }

  /** Pick one random element */
  choice<T>(arr: readonly T[]): T {
    return arr[Math.floor(this.random() * arr.length)];
  }

  /** Uniform float in [lo, hi) */
  uniform(lo: number, hi: number): number {
    return lo + this.random() * (hi - lo);
  }
}
