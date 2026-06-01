package br.com.murilovieira.fraudapi.config;

import br.com.murilovieira.fraudapi.vector.VectorStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

@Component
public class DatasetLoader {

    private static final Logger log     = LoggerFactory.getLogger(DatasetLoader.class);
    private static final String BIN_PATH = "/app/refs.bin";
    static final         int    MAGIC   = 0x52454653;
    static final         int    VERSION = 4;

    private final VectorStore vectorStore;

    public DatasetLoader(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void startLoading() {
        Thread.ofVirtual().name("dataset-loader").start(this::load);
    }

    private void load() {
        try {
            File f = new File(BIN_PATH);
            if (!f.exists()) throw new FileNotFoundException(BIN_PATH + " not found — rebuild image");
            log.info("Loading {} ({} MB)...", BIN_PATH, f.length() >> 20);
            long t0 = System.currentTimeMillis();
            loadBinary(f);
            log.info("Index ready: {} refs in {}ms", vectorStore.getRefCount(), System.currentTimeMillis() - t0);
        } catch (Exception e) {
            log.error("Failed to load dataset", e);
        }
    }

    private void loadBinary(File f) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(f), 1 << 20))) {

            int magic = dis.readInt();
            if (magic != MAGIC)
                throw new IllegalStateException("Bad magic: 0x" + Integer.toHexString(magic));
            int version = dis.readInt();
            if (version != VERSION)
                throw new IllegalStateException("Expected v" + VERSION + ", got v" + version + " — rebuild image");
            int n = dis.readInt();
            int k = dis.readInt();

            short[] centroids = readShorts(dis, k * VectorStore.DIM);
            short[] bboxMin   = readShorts(dis, k * VectorStore.DIM);
            short[] bboxMax   = readShorts(dis, k * VectorStore.DIM);
            int[]   offsets   = readInts(dis, k + 1);
            short[] rows      = readShorts(dis, n * VectorStore.DIM);
            byte[]  labels    = new byte[n];
            dis.readFully(labels);
            int[]   origIds   = readInts(dis, n);

            vectorStore.setDataset(centroids, bboxMin, bboxMax, offsets, rows, labels, origIds, k);
        }
    }

    private static short[] readShorts(DataInputStream dis, int count) throws IOException {
        short[]    out = new short[count];
        byte[]     buf = new byte[1 << 20];
        ByteBuffer bb  = ByteBuffer.wrap(buf).order(ByteOrder.BIG_ENDIAN);
        int offset = 0, remaining = count;
        while (remaining > 0) {
            int chunk = Math.min(remaining, buf.length / 2);
            dis.readFully(buf, 0, chunk * 2);
            bb.clear();
            bb.asShortBuffer().get(out, offset, chunk);
            offset += chunk;
            remaining -= chunk;
        }
        return out;
    }

    private static int[] readInts(DataInputStream dis, int count) throws IOException {
        int[]      out = new int[count];
        byte[]     buf = new byte[1 << 20];
        ByteBuffer bb  = ByteBuffer.wrap(buf).order(ByteOrder.BIG_ENDIAN);
        int offset = 0, remaining = count;
        while (remaining > 0) {
            int chunk = Math.min(remaining, buf.length / 4);
            dis.readFully(buf, 0, chunk * 4);
            bb.clear();
            bb.asIntBuffer().get(out, offset, chunk);
            offset += chunk;
            remaining -= chunk;
        }
        return out;
    }
}
