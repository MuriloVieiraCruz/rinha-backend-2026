package br.com.murilovieira.fraudapi.service;

import br.com.murilovieira.fraudapi.domain.FraudResponse;
import br.com.murilovieira.fraudapi.domain.TransactionRequest;
import br.com.murilovieira.fraudapi.vector.VectorStore;
import br.com.murilovieira.fraudapi.vector.Vectorizer;
import org.springframework.stereotype.Service;

@Service
public class FraudService {

    private static final int    K         = 5;
    private static final double THRESHOLD = 0.4;

    private final Vectorizer  vectorizer;
    private final VectorStore vectorStore;

    public FraudService(Vectorizer vectorizer, VectorStore vectorStore) {
        this.vectorizer = vectorizer;
        this.vectorStore = vectorStore;
    }

    public FraudResponse evaluate(TransactionRequest request) {
        float[] vector     = vectorizer.vectorize(request);
        int     fraudCount = vectorStore.knnFraudCount(vector, K);
        double  fraudScore = (double) fraudCount / K;
        return new FraudResponse(fraudScore < THRESHOLD, fraudScore);
    }
}
