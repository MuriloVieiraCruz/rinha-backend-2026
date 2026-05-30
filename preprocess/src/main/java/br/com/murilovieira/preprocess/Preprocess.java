package br.com.murilovieira.preprocess;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

import static java.nio.file.StandardOpenOption.*;

public final class Preprocess {

    private static final int    DIM          = 14;
    private static final int    K_CLUSTERS   = 2048;
    private static final int    K_SUPER      = 128;
    private static final int    K_ITERS      = 25;
    private static final int    QUANT_SCALE  = 10_000;
    private static final int    HEADER_BYTES = 64;
    private static final byte[] MAGIC        = {'R', 'N', 'H', '6'};
    private static final long   SEED         = 42L;

    private static float[] data;
    private static byte[]  labels;

    public static void main(String[] args) throws IOException {
        Path input  = Path.of(args.length > 0 ? args[0] : "../resources/references.json.gz");
        Path output = Path.of(args.length > 1 ? args[1] : "../resources/references.bin");
        long t0 = System.currentTimeMillis();

        System.out.println("[1/6] Lendo dataset...");
        int n = load(input);
        System.out.printf("  %,d vetores carregados%n", n);

        System.out.printf("[2/6] K-means k=%d iters=%d (k-means++ init)...%n", K_CLUSTERS, K_ITERS);
        float[] centroids   = kMeans(data, n, K_CLUSTERS, K_ITERS);
        int[]   assignment  = assign(data, n, centroids, K_CLUSTERS);

        System.out.printf("[3/6] Super-clusters ks=%d...%n", K_SUPER);
        float[] superCentroids = kMeans(centroids, K_CLUSTERS, K_SUPER, K_ITERS);
        int[]   superAssign    = assign(centroids, K_CLUSTERS, superCentroids, K_SUPER);

        System.out.println("[4/6] Layout, bbox e quantização...");
        int[]   offsets = buildOffsets(assignment, n, K_CLUSTERS);
        int[]   perm    = buildPermutation(assignment, offsets, n, K_CLUSTERS);
        short[] rows    = reorderAndQuantize(data, perm, n);
        data = null;

        short[] centroidsQ      = quantize(centroids, K_CLUSTERS);
        short[] superCentroidsQ = quantize(superCentroids, K_SUPER);

        short[] bboxMin = new short[K_CLUSTERS * DIM];
        short[] bboxMax = new short[K_CLUSTERS * DIM];
        computeBboxes(rows, offsets, K_CLUSTERS, bboxMin, bboxMax);

        int[]   superOffsets  = buildSuperOffsets(superAssign, K_SUPER, K_CLUSTERS);
        int[]   superClusters = buildSuperClusters(superAssign, superOffsets, K_SUPER, K_CLUSTERS);
        short[] superBboxMin  = new short[K_SUPER * DIM];
        short[] superBboxMax  = new short[K_SUPER * DIM];
        computeSuperBboxes(bboxMin, bboxMax, superClusters, superOffsets, K_SUPER, superBboxMin, superBboxMax);

        System.out.println("[5/6] Reordenando labels...");
        byte[] labelsReordered = new byte[n];
        for (int i = 0; i < n; i++) labelsReordered[i] = labels[perm[i]];
        labels = null;

        System.out.println("[6/6] Escrevendo RNH6...");
        Path tmp = output.resolveSibling(output.getFileName() + ".tmp");
        write(tmp, n, K_CLUSTERS, K_SUPER,
              offsets, centroidsQ, bboxMin, bboxMax,
              superOffsets, superClusters, superCentroidsQ, superBboxMin, superBboxMax,
              rows, labelsReordered);
        Files.move(tmp, output, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);

        System.out.printf("Concluído em %.1fs → %s%n",
                (System.currentTimeMillis() - t0) / 1000.0, output.toAbsolutePath());
    }

    // ── Leitura do dataset ────────────────────────────────────────────────────

    private static int load(Path input) throws IOException {
        int cap = 3_200_000;
        data   = new float[cap * DIM];
        labels = new byte[cap];
        int n  = 0;

        try (InputStream raw = Files.newInputStream(input);
             GZIPInputStream gz = new GZIPInputStream(raw, 64 * 1024);
             var p = new JsonFactory().createParser(gz)) {

            if (p.nextToken() != JsonToken.START_ARRAY) throw new IOException("JSON array esperado");

            while (p.nextToken() == JsonToken.START_OBJECT) {
                if (n >= cap) {
                    cap = cap + (cap >> 1);
                    data   = Arrays.copyOf(data,   cap * DIM);
                    labels = Arrays.copyOf(labels, cap);
                }
                int    base = n * DIM;
                int    vi   = 0;
                byte   lbl  = -1;
                while (p.nextToken() != JsonToken.END_OBJECT) {
                    String field = p.currentName();
                    p.nextToken();
                    if ("vector".equals(field)) {
                        while (p.nextToken() != JsonToken.END_ARRAY)
                            data[base + vi++] = p.getFloatValue();
                    } else if ("label".equals(field)) {
                        lbl = "fraud".equals(p.getText()) ? (byte) 1 : (byte) 0;
                    } else {
                        p.skipChildren();
                    }
                }
                if (vi != DIM || lbl < 0) throw new IOException("Registro inválido em índice " + n);
                labels[n++] = lbl;
                if (n % 500_000 == 0) System.out.printf("  ... %,d%n", n);
            }
        }
        data   = Arrays.copyOf(data,   n * DIM);
        labels = Arrays.copyOf(labels, n);
        return n;
    }

    // ── K-means com k-means++ e atribuição paralela ───────────────────────────

    private static float[] kMeans(float[] data, int n, int k, int iters) {
        Random rng = new Random(SEED);
        float[] centroids = kMeansPlusPlusInit(data, n, k, rng);

        int[]   assignment = new int[n];
        float[] sums       = new float[k * DIM];
        int[]   counts     = new int[k];

        for (int iter = 0; iter < iters; iter++) {
            assignParallel(data, n, centroids, k, assignment);
            Arrays.fill(sums, 0f);
            Arrays.fill(counts, 0);
            for (int i = 0; i < n; i++) {
                int c = assignment[i], cb = c * DIM, db = i * DIM;
                counts[c]++;
                for (int d = 0; d < DIM; d++) sums[cb + d] += data[db + d];
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    int cb = c * DIM; float inv = 1f / counts[c];
                    for (int d = 0; d < DIM; d++) centroids[cb + d] = sums[cb + d] * inv;
                } else {
                    System.arraycopy(data, rng.nextInt(n) * DIM, centroids, c * DIM, DIM);
                }
            }
            System.out.printf("  iter %d/%d%n", iter + 1, iters);
        }
        return centroids;
    }

    private static float[] kMeansPlusPlusInit(float[] data, int n, int k, Random rng) {
        float[] centroids = new float[k * DIM];
        int     first     = rng.nextInt(n);
        System.arraycopy(data, first * DIM, centroids, 0, DIM);

        float[] minDist = new float[n];
        Arrays.fill(minDist, Float.MAX_VALUE);

        for (int c = 1; c < k; c++) {
            int prevBase = (c - 1) * DIM;
            double total = 0;
            for (int i = 0; i < n; i++) {
                float d = distSq(data, i, centroids, c - 1, prevBase);
                if (d < minDist[i]) minDist[i] = d;
                total += minDist[i];
            }
            double target = rng.nextDouble() * total;
            int chosen = 0;
            for (int i = 0; i < n; i++) {
                target -= minDist[i];
                if (target <= 0) { chosen = i; break; }
            }
            System.arraycopy(data, chosen * DIM, centroids, c * DIM, DIM);
        }
        return centroids;
    }

    private static void assignParallel(float[] data, int n, float[] centroids, int k, int[] out) {
        IntStream.range(0, n).parallel().forEach(i -> {
            float best = Float.MAX_VALUE;
            int   bi   = 0;
            for (int c = 0; c < k; c++) {
                float d = distSq(data, i, centroids, c, c * DIM);
                if (d < best) { best = d; bi = c; }
            }
            out[i] = bi;
        });
    }

    private static int[] assign(float[] data, int n, float[] centroids, int k) {
        int[] out = new int[n];
        assignParallel(data, n, centroids, k, out);
        return out;
    }

    private static float distSq(float[] a, int ai, float[] b, int bi, int bBase) {
        int   aBase = ai * DIM;
        float sum   = 0;
        for (int d = 0; d < DIM; d++) { float diff = a[aBase + d] - b[bBase + d]; sum += diff * diff; }
        return sum;
    }

    // ── Layout e quantização ──────────────────────────────────────────────────

    private static int[] buildOffsets(int[] assignment, int n, int k) {
        int[] counts = new int[k];
        for (int i = 0; i < n; i++) counts[assignment[i]]++;
        int[] offsets = new int[k + 1];
        for (int c = 0; c < k; c++) offsets[c + 1] = offsets[c] + counts[c];
        return offsets;
    }

    private static int[] buildPermutation(int[] assignment, int[] offsets, int n, int k) {
        int[] pos  = Arrays.copyOf(offsets, k);
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[pos[assignment[i]]++] = i;
        return perm;
    }

    private static short[] reorderAndQuantize(float[] data, int[] perm, int n) {
        short[] rows = new short[n * DIM];
        for (int i = 0; i < n; i++) {
            int src = perm[i] * DIM, dst = i * DIM;
            for (int d = 0; d < DIM; d++) {
                int q = Math.round(data[src + d] * QUANT_SCALE);
                rows[dst + d] = (short) Math.max(-QUANT_SCALE, Math.min(QUANT_SCALE, q));
            }
        }
        return rows;
    }

    private static short[] quantize(float[] centroids, int k) {
        short[] out = new short[k * DIM];
        for (int i = 0; i < k * DIM; i++) {
            int q = Math.round(centroids[i] * QUANT_SCALE);
            out[i] = (short) Math.max(-QUANT_SCALE, Math.min(QUANT_SCALE, q));
        }
        return out;
    }

    // ── Bboxes ────────────────────────────────────────────────────────────────

    private static void computeBboxes(short[] rows, int[] offsets, int k,
                                      short[] bboxMin, short[] bboxMax) {
        Arrays.fill(bboxMin, Short.MAX_VALUE);
        Arrays.fill(bboxMax, Short.MIN_VALUE);
        for (int c = 0; c < k; c++) {
            int base = c * DIM;
            for (int i = offsets[c]; i < offsets[c + 1]; i++) {
                int rb = i * DIM;
                for (int d = 0; d < DIM; d++) {
                    if (rows[rb + d] < bboxMin[base + d]) bboxMin[base + d] = rows[rb + d];
                    if (rows[rb + d] > bboxMax[base + d]) bboxMax[base + d] = rows[rb + d];
                }
            }
            if (offsets[c] == offsets[c + 1]) {
                Arrays.fill(bboxMin, base, base + DIM, (short) 0);
                Arrays.fill(bboxMax, base, base + DIM, (short) 0);
            }
        }
    }

    private static int[] buildSuperOffsets(int[] superAssign, int ks, int k) {
        int[] counts  = new int[ks];
        for (int c = 0; c < k; c++) counts[superAssign[c]]++;
        int[] offsets = new int[ks + 1];
        for (int s = 0; s < ks; s++) offsets[s + 1] = offsets[s] + counts[s];
        return offsets;
    }

    private static int[] buildSuperClusters(int[] superAssign, int[] superOffsets, int ks, int k) {
        int[] pos           = Arrays.copyOf(superOffsets, ks);
        int[] superClusters = new int[k];
        for (int c = 0; c < k; c++) superClusters[pos[superAssign[c]]++] = c;
        return superClusters;
    }

    private static void computeSuperBboxes(short[] bboxMin, short[] bboxMax,
                                           int[] superClusters, int[] superOffsets, int ks,
                                           short[] superBboxMin, short[] superBboxMax) {
        Arrays.fill(superBboxMin, Short.MAX_VALUE);
        Arrays.fill(superBboxMax, Short.MIN_VALUE);
        for (int s = 0; s < ks; s++) {
            int sb = s * DIM;
            for (int i = superOffsets[s]; i < superOffsets[s + 1]; i++) {
                int cb = superClusters[i] * DIM;
                for (int d = 0; d < DIM; d++) {
                    if (bboxMin[cb + d] < superBboxMin[sb + d]) superBboxMin[sb + d] = bboxMin[cb + d];
                    if (bboxMax[cb + d] > superBboxMax[sb + d]) superBboxMax[sb + d] = bboxMax[cb + d];
                }
            }
            if (superOffsets[s] == superOffsets[s + 1]) {
                Arrays.fill(superBboxMin, sb, sb + DIM, (short) 0);
                Arrays.fill(superBboxMax, sb, sb + DIM, (short) 0);
            }
        }
    }

    // ── Escrita do binário RNH6 ───────────────────────────────────────────────

    private static void write(Path out, int n, int k, int ks,
                              int[] offsets, short[] centroids, short[] bboxMin, short[] bboxMax,
                              int[] superOffsets, int[] superClusters,
                              short[] superCentroids, short[] superBboxMin, short[] superBboxMax,
                              short[] rows, byte[] labels) throws IOException {
        try (FileChannel ch = FileChannel.open(out, CREATE, WRITE, TRUNCATE_EXISTING)) {
            ByteBuffer hdr = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            hdr.put(MAGIC).putInt(n).putInt(k).putInt(ks).putInt(DIM).putInt(QUANT_SCALE);
            hdr.position(HEADER_BYTES).flip();
            ch.write(hdr);

            writeInts(ch, offsets);
            writeShorts(ch, centroids);
            writeShorts(ch, bboxMin);
            writeShorts(ch, bboxMax);
            writeInts(ch, superOffsets);
            writeInts(ch, superClusters);
            writeShorts(ch, superCentroids);
            writeShorts(ch, superBboxMin);
            writeShorts(ch, superBboxMax);
            writeShorts(ch, rows);
            ch.write(ByteBuffer.wrap(labels));

            System.out.printf("  %,d bytes escritos%n", ch.size());
        }
    }

    private static void writeShorts(FileChannel ch, short[] arr) throws IOException {
        ByteBuffer buf = ByteBuffer.allocateDirect(65536).order(ByteOrder.LITTLE_ENDIAN);
        for (short s : arr) {
            if (!buf.hasRemaining()) { buf.flip(); ch.write(buf); buf.clear(); }
            buf.putShort(s);
        }
        if (buf.position() > 0) { buf.flip(); ch.write(buf); }
    }

    private static void writeInts(FileChannel ch, int[] arr) throws IOException {
        ByteBuffer buf = ByteBuffer.allocateDirect(65536).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : arr) {
            if (buf.remaining() < Integer.BYTES) { buf.flip(); ch.write(buf); buf.clear(); }
            buf.putInt(v);
        }
        if (buf.position() > 0) { buf.flip(); ch.write(buf); }
    }
}
