package br.com.murilovieira.preprocess;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonToken;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public final class Preprocess {

    static final int    DIM         = 14;
    static final int    K           = 2048;
    static final int    MAX_CLUSTER = 1024;
    static final int    ITERS       = 12;
    static final int    SAMPLE      = 200_000;
    static final int    MAGIC       = 0x52454653;
    static final int    VERSION     = 4;
    static final int    SCALE       = 10_000;
    static final long   SEED        = 0x9E3779B97F4A7C15L;

    public static void main(String[] args) throws Exception {
        String inputPath  = args.length > 0 ? args[0] : "../resources/references.json.gz";
        String outputPath = args.length > 1 ? args[1] : "../resources/refs.bin";

        long t0 = System.currentTimeMillis();

        System.out.println("[1/5] Lendo dataset e quantizando...");
        int MAX = 3_100_000;
        short[] flat   = new short[MAX * DIM];
        byte[]  labels = new byte[MAX];
        int n = load(inputPath, flat, labels);
        System.out.printf("  %,d vetores carregados%n", n);

        System.out.printf("[2/5] K-means k=%d em amostra de %,d (k-means++, %d iters)...%n", K, SAMPLE, ITERS);
        Random rng = new Random(SEED);
        short[] centroids = kmeans(sample(flat, n, SAMPLE, rng), Math.min(K, n), rng);
        int k = centroids.length / DIM;

        int[] coarse = assignAll(flat, n, centroids, k);

        System.out.println("[3/5] Dividindo clusters grandes...");
        int[][] inverted  = invert(coarse, n, k);
        int[]   finalAssign = new int[n];
        int     kFinal    = 0, splits = 0;

        for (int c = 0; c < k; c++) {
            int[] members = inverted[c];
            if (members.length <= MAX_CLUSTER) {
                int id = kFinal++;
                for (int p : members) finalAssign[p] = id;
            } else {
                splits++;
                short[] sub  = gather(flat, members);
                int     subK = (members.length + MAX_CLUSTER - 1) / MAX_CLUSTER;
                short[] subC = kmeans(sub, subK, new Random(SEED + c + 1));
                int     subKA = subC.length / DIM;
                int[]   subA = assignAll(sub, members.length, subC, subKA);
                int[]   subIds = new int[subKA];
                for (int s = 0; s < subKA; s++) subIds[s] = kFinal++;
                for (int j = 0; j < members.length; j++) finalAssign[members[j]] = subIds[subA[j]];
            }
        }
        System.out.printf("  %d coarse → %d clusters finais (%d divididos)%n", k, kFinal, splits);

        System.out.println("[4/5] Calculando offsets, centroides, bboxes...");
        int[] offsets = new int[kFinal + 1];
        for (int i = 0; i < n; i++) offsets[finalAssign[i] + 1]++;
        for (int c = 0; c < kFinal; c++) offsets[c + 1] += offsets[c];

        short[] rows     = new short[n * DIM];
        byte[]  outLab   = new byte[n];
        int[]   origIds  = new int[n];
        short[] cOut     = new short[kFinal * DIM];
        short[] bMin     = new short[kFinal * DIM];
        short[] bMax     = new short[kFinal * DIM];
        long[]  acc      = new long[kFinal * DIM];

        for (int c = 0; c < kFinal; c++) {
            int cb = c * DIM;
            Arrays.fill(bMin, cb, cb + DIM, Short.MAX_VALUE);
            Arrays.fill(bMax, cb, cb + DIM, Short.MIN_VALUE);
        }

        int[] cursor = Arrays.copyOf(offsets, kFinal);
        for (int i = 0; i < n; i++) {
            int c   = finalAssign[i];
            int pos = cursor[c]++;
            int src = i * DIM, dst = pos * DIM, cb = c * DIM;
            for (int d = 0; d < DIM; d++) {
                short v = flat[src + d];
                rows[dst + d] = v;
                if (v < bMin[cb + d]) bMin[cb + d] = v;
                if (v > bMax[cb + d]) bMax[cb + d] = v;
                acc[cb + d] += v;
            }
            outLab[pos]  = labels[i];
            origIds[pos] = i;
        }
        for (int c = 0; c < kFinal; c++) {
            int cnt = offsets[c + 1] - offsets[c];
            if (cnt == 0) continue;
            int cb = c * DIM;
            for (int d = 0; d < DIM; d++) cOut[cb + d] = (short) Math.round((double) acc[cb + d] / cnt);
        }

        System.out.printf("[5/5] Escrevendo %s...%n", outputPath);
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(outputPath), 1 << 23))) {
            dos.writeInt(MAGIC);
            dos.writeInt(VERSION);
            dos.writeInt(n);
            dos.writeInt(kFinal);
            writeShorts(dos, cOut);
            writeShorts(dos, bMin);
            writeShorts(dos, bMax);
            writeInts(dos, offsets);
            writeShorts(dos, rows);
            dos.write(outLab);
            writeInts(dos, origIds);
        }

        long size = new File(outputPath).length();
        System.out.printf("Concluído em %.1fs → %,d bytes (%.1f MB)%n",
                (System.currentTimeMillis() - t0) / 1000.0, size, size / 1_048_576.0);
    }

    private static int load(String path, short[] flat, byte[] labels) throws IOException {
        int n = 0;
        try (InputStream raw = new FileInputStream(path);
             GZIPInputStream gz = new GZIPInputStream(raw, 64 * 1024);
             var p = new JsonFactory().createParser(gz)) {

            if (p.nextToken() != JsonToken.START_ARRAY) throw new IOException("JSON array esperado");

            while (p.nextToken() == JsonToken.START_OBJECT) {
                int base = n * DIM, vi = 0;
                byte lbl = -1;
                while (p.nextToken() != JsonToken.END_OBJECT) {
                    String field = p.currentName();
                    p.nextToken();
                    if ("vector".equals(field)) {
                        while (p.nextToken() != JsonToken.END_ARRAY) {
                            int q = Math.round(p.getFloatValue() * SCALE);
                            flat[base + vi++] = (short) (q > SCALE ? SCALE : q < -SCALE ? -SCALE : q);
                        }
                    } else if ("label".equals(field)) {
                        lbl = "fraud".equals(p.getText()) ? (byte) 1 : 0;
                    } else {
                        p.skipChildren();
                    }
                }
                labels[n++] = lbl;
                if (n % 500_000 == 0) System.out.printf("  ... %,d%n", n);
            }
        }
        return n;
    }

    static short[] kmeans(short[] data, int k, Random rng) {
        int m = data.length / DIM;
        k = Math.min(k, m);
        short[] cents  = kmeansPlusPlus(data, m, k, rng);
        int[]   assign = new int[m];
        for (int iter = 0; iter < ITERS; iter++) {
            assignInto(data, m, cents, k, assign);
            long[] acc = new long[k * DIM];
            int[]  cnt = new int[k];
            for (int i = 0; i < m; i++) {
                int c = assign[i], src = i * DIM, cb = c * DIM;
                cnt[c]++;
                for (int d = 0; d < DIM; d++) acc[cb + d] += data[src + d];
            }
            for (int c = 0; c < k; c++) {
                if (cnt[c] == 0) continue;
                int cb = c * DIM;
                for (int d = 0; d < DIM; d++)
                    cents[cb + d] = (short) Math.round((double) acc[cb + d] / cnt[c]);
            }
        }
        return cents;
    }

    private static short[] kmeansPlusPlus(short[] data, int m, int k, Random rng) {
        short[] cents = new short[k * DIM];
        long[]  dist  = new long[m];
        int first = rng.nextInt(m);
        System.arraycopy(data, first * DIM, cents, 0, DIM);
        for (int i = 0; i < m; i++) dist[i] = sqDist(data, i * DIM, cents, 0);

        for (int c = 1; c < k; c++) {
            double total = 0;
            for (long dd : dist) total += dd;
            double threshold = rng.nextDouble() * total, cum = 0;
            int chosen = m - 1;
            for (int i = 0; i < m; i++) { cum += dist[i]; if (cum >= threshold) { chosen = i; break; } }
            int cb = c * DIM;
            System.arraycopy(data, chosen * DIM, cents, cb, DIM);
            for (int i = 0; i < m; i++) {
                long d = sqDist(data, i * DIM, cents, cb);
                if (d < dist[i]) dist[i] = d;
            }
        }
        return cents;
    }

    static void assignInto(short[] data, int m, short[] cents, int k, int[] out) {
        IntStream.range(0, m).parallel().forEach(i -> out[i] = nearest(data, i * DIM, cents, k));
    }

    static int[] assignAll(short[] data, int m, short[] cents, int k) {
        int[] out = new int[m];
        assignInto(data, m, cents, k, out);
        return out;
    }

    private static int nearest(short[] data, int off, short[] cents, int k) {
        int best = 0;
        long bestD = Long.MAX_VALUE;
        for (int c = 0; c < k; c++) {
            long d = sqDist(data, off, cents, c * DIM);
            if (d < bestD) { bestD = d; best = c; }
        }
        return best;
    }

    static long sqDist(short[] a, int ao, short[] b, int bo) {
        long s = 0;
        for (int d = 0; d < DIM; d++) { int e = a[ao + d] - b[bo + d]; s += (long) e * e; }
        return s;
    }

    private static short[] sample(short[] flat, int n, int s, Random rng) {
        if (s >= n) return Arrays.copyOf(flat, n * DIM);
        short[] out = new short[s * DIM];
        for (int i = 0; i < s; i++) {
            int src = rng.nextInt(n) * DIM;
            System.arraycopy(flat, src, out, i * DIM, DIM);
        }
        return out;
    }

    private static int[][] invert(int[] assign, int n, int k) {
        int[] sizes = new int[k];
        for (int i = 0; i < n; i++) sizes[assign[i]]++;
        int[][] out = new int[k][];
        for (int c = 0; c < k; c++) out[c] = new int[sizes[c]];
        int[] pos = new int[k];
        for (int i = 0; i < n; i++) { int c = assign[i]; out[c][pos[c]++] = i; }
        return out;
    }

    private static short[] gather(short[] flat, int[] ids) {
        short[] out = new short[ids.length * DIM];
        for (int j = 0; j < ids.length; j++)
            System.arraycopy(flat, ids[j] * DIM, out, j * DIM, DIM);
        return out;
    }

    private static void writeShorts(DataOutputStream dos, short[] arr) throws IOException {
        byte[]     buf = new byte[1 << 22];
        ByteBuffer bb  = ByteBuffer.wrap(buf).order(ByteOrder.BIG_ENDIAN);
        int remaining = arr.length, offset = 0;
        while (remaining > 0) {
            int chunk = Math.min(remaining, buf.length / 2);
            bb.clear();
            bb.asShortBuffer().put(arr, offset, chunk);
            dos.write(buf, 0, chunk * 2);
            offset += chunk;
            remaining -= chunk;
        }
    }

    private static void writeInts(DataOutputStream dos, int[] arr) throws IOException {
        byte[]     buf = new byte[1 << 22];
        ByteBuffer bb  = ByteBuffer.wrap(buf).order(ByteOrder.BIG_ENDIAN);
        int remaining = arr.length, offset = 0;
        while (remaining > 0) {
            int chunk = Math.min(remaining, buf.length / 4);
            bb.clear();
            bb.asIntBuffer().put(arr, offset, chunk);
            dos.write(buf, 0, chunk * 4);
            offset += chunk;
            remaining -= chunk;
        }
    }
}
