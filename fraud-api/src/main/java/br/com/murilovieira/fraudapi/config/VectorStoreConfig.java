package br.com.murilovieira.fraudapi.config;

import br.com.murilovieira.fraudapi.vector.VectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

@Configuration
public class VectorStoreConfig {

    private static final byte[] MAGIC        = {'R', 'N', 'H', '6'};
    private static final int    HEADER_BYTES = 64;
    private static final int    TOP_K        = 5;

    @Bean
    public VectorStore vectorStore(AppProperties props) throws IOException {
        Path path = Path.of(props.referencesPath());

        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            var raw = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
            raw.order(ByteOrder.LITTLE_ENDIAN);

            for (int i = 0; i < MAGIC.length; i++) {
                if (raw.get(i) != MAGIC[i])
                    throw new IOException("Índice inválido: magic incorreto em " + path);
            }

            int n   = raw.getInt(4);
            int k   = raw.getInt(8);
            int ks  = raw.getInt(12);
            int dim = raw.getInt(16);
            if (dim != VectorStore.DIM)
                throw new IOException("DIM inesperado: " + dim + " (esperado " + VectorStore.DIM + ")");

            long pos = HEADER_BYTES;

            int[]   offsets        = readInts(raw, pos, k + 1);   pos += (long)(k + 1)  * Integer.BYTES;
            short[] centroids      = readShorts(raw, pos, k * dim); pos += (long) k  * dim * Short.BYTES;
            short[] bboxMin        = readShorts(raw, pos, k * dim); pos += (long) k  * dim * Short.BYTES;
            short[] bboxMax        = readShorts(raw, pos, k * dim); pos += (long) k  * dim * Short.BYTES;
            int[]   superOffsets   = readInts(raw, pos, ks + 1);  pos += (long)(ks + 1) * Integer.BYTES;
            int[]   superClusters  = readInts(raw, pos, k);        pos += (long) k  * Integer.BYTES;
            short[] superCentroids = readShorts(raw, pos, ks * dim); pos += (long) ks * dim * Short.BYTES;
            short[] superBboxMin   = readShorts(raw, pos, ks * dim); pos += (long) ks * dim * Short.BYTES;
            short[] superBboxMax   = readShorts(raw, pos, ks * dim); pos += (long) ks * dim * Short.BYTES;

            short[] rows = readRowsStream(ch, pos, n, dim);
            pos += (long) n * dim * Short.BYTES;

            byte[] labels = new byte[n];
            raw.get((int) pos, labels);

            return new VectorStore(n, k, ks,
                    offsets, centroids, bboxMin, bboxMax,
                    superOffsets, superClusters,
                    superCentroids, superBboxMin, superBboxMax,
                    rows, labels, TOP_K);
        }
    }

    private static short[] readRowsStream(FileChannel ch, long startPos, int n, int dim) throws IOException {
        short[]    rows = new short[n * dim];
        ByteBuffer buf  = ByteBuffer.allocate(65536).order(ByteOrder.LITTLE_ENDIAN);
        int idx = 0;
        long pos = startPos;
        while (idx < rows.length) {
            buf.clear();
            int read = ch.read(buf, pos);
            if (read < 0) throw new IOException("EOF inesperado lendo rows");
            buf.flip();
            pos += read;
            while (buf.hasRemaining() && idx < rows.length) rows[idx++] = buf.getShort();
        }
        return rows;
    }

    private static int[] readInts(ByteBuffer buf, long pos, int count) {
        int[] arr = new int[count];
        for (int i = 0; i < count; i++) arr[i] = buf.getInt((int)(pos + (long) i * Integer.BYTES));
        return arr;
    }

    private static short[] readShorts(ByteBuffer buf, long pos, int count) {
        short[] arr = new short[count];
        for (int i = 0; i < count; i++) arr[i] = buf.getShort((int)(pos + (long) i * Short.BYTES));
        return arr;
    }
}
