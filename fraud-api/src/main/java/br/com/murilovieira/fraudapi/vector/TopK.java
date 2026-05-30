package br.com.murilovieira.fraudapi.vector;

final class TopK {

    private final long[] dist;
    private final int[]  idx;
    private final int    cap;
    private int          size;

    TopK(int cap) {
        this.cap  = cap;
        this.dist = new long[cap];
        this.idx  = new int[cap];
    }

    void reset()     { size = 0; }
    boolean isFull() { return size == cap; }
    long worst()     { return dist[0]; }

    void offer(long d, int i) {
        if (size < cap) {
            dist[size] = d;
            idx[size]  = i;
            siftUp(size++);
        } else if (d < dist[0]) {
            dist[0] = d;
            idx[0]  = i;
            siftDown(0);
        }
    }

    int countFrauds(byte[] labels) {
        int count = 0;
        for (int i = 0; i < size; i++) count += labels[idx[i]];
        return count;
    }

    private void siftUp(int i) {
        while (i > 0) {
            int p = (i - 1) >>> 1;
            if (dist[i] > dist[p]) { swap(i, p); i = p; } else break;
        }
    }

    private void siftDown(int i) {
        while (true) {
            int l = 2*i+1, r = 2*i+2, best = i;
            if (l < size && dist[l] > dist[best]) best = l;
            if (r < size && dist[r] > dist[best]) best = r;
            if (best != i) { swap(i, best); i = best; } else break;
        }
    }

    private void swap(int a, int b) {
        long td = dist[a]; dist[a] = dist[b]; dist[b] = td;
        int  ti = idx[a];  idx[a]  = idx[b];  idx[b]  = ti;
    }
}
