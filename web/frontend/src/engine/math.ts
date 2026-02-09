/**
 * Math utilities: normal CDF, erf, statistics, correlation.
 */

export function erf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y =
    1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

export function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

export function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

export function std(arr: number[], ddof = 0): number {
  if (arr.length <= ddof) return 0;
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / (arr.length - ddof));
}

export function cumMax(arr: number[]): number[] {
  const result: number[] = [];
  let mx = -Infinity;
  for (const v of arr) {
    mx = Math.max(mx, v);
    result.push(mx);
  }
  return result;
}

/** Pearson correlation between two arrays of equal length. */
export function corr(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  const ma = mean(a.slice(0, n));
  const mb = mean(b.slice(0, n));
  let sumAB = 0,
    sumAA = 0,
    sumBB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    sumAB += da * db;
    sumAA += da * da;
    sumBB += db * db;
  }
  const denom = Math.sqrt(sumAA * sumBB);
  return denom > 0 ? sumAB / denom : 0;
}

/** Full correlation matrix for rows of returns. */
export function corrMatrix(returnRows: number[][]): number[][] {
  const n = returnRows.length;
  const mat: number[][] = [];
  for (let i = 0; i < n; i++) {
    mat.push([]);
    for (let j = 0; j < n; j++) {
      if (i === j) {
        mat[i].push(1);
      } else if (j < i) {
        mat[i].push(mat[j][i]);
      } else {
        mat[i].push(Math.round(corr(returnRows[i], returnRows[j]) * 100) / 100);
      }
    }
  }
  return mat;
}
