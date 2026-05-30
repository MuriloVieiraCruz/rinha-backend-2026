package br.com.murilovieira.fraudapi.vector;

public final class VectorStore {

    public static final int DIM         = Vectorizer.DIM;
    static final int QUANT_SCALE = 10_000;

    private final int     n, k, ks;
    private final int[]   offsets;
    private final short[] centroids;
    private final short[] bboxMin;
    private final short[] bboxMax;
    private final int[]   superOffsets;
    private final int[]   superClusters;
    private final short[] superCentroids;
    private final short[] superBboxMin;
    private final short[] superBboxMax;
    private final short[] rows;
    private final byte[]  labels;

    private final ThreadLocal<int[]> queryTL = ThreadLocal.withInitial(() -> new int[DIM]);
    private final ThreadLocal<TopK>  topKTL;

    public VectorStore(int n, int k, int ks,
                       int[] offsets,
                       short[] centroids, short[] bboxMin, short[] bboxMax,
                       int[] superOffsets, int[] superClusters,
                       short[] superCentroids, short[] superBboxMin, short[] superBboxMax,
                       short[] rows, byte[] labels,
                       int topKCap) {
        this.n              = n;
        this.k              = k;
        this.ks             = ks;
        this.offsets        = offsets;
        this.centroids      = centroids;
        this.bboxMin        = bboxMin;
        this.bboxMax        = bboxMax;
        this.superOffsets   = superOffsets;
        this.superClusters  = superClusters;
        this.superCentroids = superCentroids;
        this.superBboxMin   = superBboxMin;
        this.superBboxMax   = superBboxMax;
        this.rows           = rows;
        this.labels         = labels;
        this.topKTL         = ThreadLocal.withInitial(() -> new TopK(topKCap));
    }

    public int knnFraudCount(float[] query, int kNeighbors) {
        int[] q = queryTL.get();
        for (int i = 0; i < DIM; i++) {
            int v = Math.round(query[i] * QUANT_SCALE);
            q[i] = v < -QUANT_SCALE ? -QUANT_SCALE : v > QUANT_SCALE ? QUANT_SCALE : v;
        }

        TopK top = topKTL.get();
        top.reset();

        int seedSuper   = nearestSuperCluster(q);
        int seedCluster = nearestClusterInSuper(q, seedSuper);
        scanCluster(seedCluster, q, top);

        for (int si = 0; si < ks; si++) {
            if (top.isFull() && bboxLowerBound(superBboxMin, superBboxMax, si, q) > top.worst())
                continue;
            for (int ci = superOffsets[si]; ci < superOffsets[si + 1]; ci++) {
                int cluster = superClusters[ci];
                if (cluster == seedCluster) continue;
                if (top.isFull() && bboxLowerBound(bboxMin, bboxMax, cluster, q) > top.worst())
                    continue;
                scanCluster(cluster, q, top);
            }
        }

        return top.countFrauds(labels);
    }

    private int nearestSuperCluster(int[] q) {
        long best = Long.MAX_VALUE;
        int  bi   = 0;
        for (int s = 0; s < ks; s++) {
            long d = centroidDistSq(superCentroids, s, q);
            if (d < best) { best = d; bi = s; }
        }
        return bi;
    }

    private int nearestClusterInSuper(int[] q, int si) {
        long best    = Long.MAX_VALUE;
        int  bestCluster = superClusters[superOffsets[si]];
        for (int ci = superOffsets[si]; ci < superOffsets[si + 1]; ci++) {
            int  cluster = superClusters[ci];
            long d       = centroidDistSq(centroids, cluster, q);
            if (d < best) { best = d; bestCluster = cluster; }
        }
        return bestCluster;
    }

    private static long centroidDistSq(short[] cents, int idx, int[] q) {
        int  base = idx * DIM;
        long sum  = 0;
        for (int d = 0; d < DIM; d++) {
            long diff = q[d] - cents[base + d];
            sum += diff * diff;
        }
        return sum;
    }

    private static long bboxLowerBound(short[] min, short[] max, int idx, int[] q) {
        int  base = idx * DIM;
        long sum  = 0;
        for (int d = 0; d < DIM; d++) {
            int  qi  = q[d];
            long gap = qi < min[base + d] ? min[base + d] - qi
                     : qi > max[base + d] ? qi - max[base + d]
                     : 0;
            sum += gap * gap;
        }
        return sum;
    }

    private void scanCluster(int cluster, int[] q, TopK top) {
        final short[] vb   = rows;
        final int     from = offsets[cluster];
        final int     to   = offsets[cluster + 1];
        for (int i = from; i < to; i++) {
            final long worst = top.isFull() ? top.worst() : Long.MAX_VALUE;
            final int  base  = i * DIM;
            long d, sum;
            d = q[0]  - vb[base];      sum  = d*d; if (sum > worst) continue;
            d = q[1]  - vb[base + 1];  sum += d*d; if (sum > worst) continue;
            d = q[2]  - vb[base + 2];  sum += d*d; if (sum > worst) continue;
            d = q[3]  - vb[base + 3];  sum += d*d; if (sum > worst) continue;
            d = q[4]  - vb[base + 4];  sum += d*d; if (sum > worst) continue;
            d = q[5]  - vb[base + 5];  sum += d*d; if (sum > worst) continue;
            d = q[6]  - vb[base + 6];  sum += d*d; if (sum > worst) continue;
            d = q[7]  - vb[base + 7];  sum += d*d; if (sum > worst) continue;
            d = q[8]  - vb[base + 8];  sum += d*d; if (sum > worst) continue;
            d = q[9]  - vb[base + 9];  sum += d*d; if (sum > worst) continue;
            d = q[10] - vb[base + 10]; sum += d*d; if (sum > worst) continue;
            d = q[11] - vb[base + 11]; sum += d*d; if (sum > worst) continue;
            d = q[12] - vb[base + 12]; sum += d*d; if (sum > worst) continue;
            d = q[13] - vb[base + 13]; sum += d*d;
            top.offer(sum, i);
        }
    }

}
